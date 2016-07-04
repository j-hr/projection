from __future__ import print_function
from dolfin import Function, VectorFunctionSpace, FunctionSpace, assemble, Expression, CellSize, DOLFIN_EPS, parameters, \
    plot
from dolfin.cpp.common import info, begin, end
from dolfin.cpp.function import FunctionAssigner
from dolfin.cpp.la import LUSolver, KrylovSolver, as_backend_type, VectorSpaceBasis, Vector, PETScKrylovSolver, \
    PETScOptions
from dolfin.functions import TrialFunction, TestFunction, Constant, FacetNormal
from ufl import dot, dx, grad, system, div, inner, sym, Identity, transpose, nabla_grad, sqrt, min_value

import general_solver as gs

# QQ split rotation, lagrange scheme?
# (implement as this Solver subclass? Can class be subclass of class with same name?)
# in current state almost no code is saved by subclassing this, one must rewrite whole solve()


class Solver(gs.GeneralSolver):
    def __init__(self, args, tc, metadata):
        gs.GeneralSolver.__init__(self, args, tc, metadata)
        self.metadata['hasTentativeV'] = True

        self.solver_vel_tent = None
        self.solver_vel_cor = None
        self.solver_p = None
        self.solver_rot = None
        self.null_space = None

        # input parameters
        self.bc = args.bc
        self.forceOutflow = args.fo
        self.useLaplace = args.laplace
        self.use_full_SUPG = args.cs
        self.bcv = 'NOT' if self.useLaplace else args.bcv
        if self.bcv == 'CDN':
            info('Using classical do nothing condition (Tn=0).')
        if self.bcv == 'DDN':
            info('Using directional do nothing condition (Tn=0.5*negative(u.n)u).')
        if self.bcv == 'LAP':
            info('Using laplace neutral condition (grad(u)n=0).')
        self.stabCoef = args.stab
        self.stabilize = (args.stab > DOLFIN_EPS)
        if self.stabilize:
            if self.use_full_SUPG:
                info('Used consistent streamline-diffusion stabilization with coef.: %f' % args.stab)
            else:
                info('Used non-consistent streamline-diffusion stabilization with coef.: %f' % args.stab)
        else:
            info('No stabilization used.')
        self.solvers = args.solvers
        self.useRotationScheme = args.r
        self.metadata['hasTentativeP'] = self.useRotationScheme

        self.B = args.B
        self.use_ema = args.ema
        self.cbcDelta = args.cbcDelta
        self.prec_v = args.precV
        self.prec_p = args.precP
        self.precision_rel_v_tent = args.prv1
        self.precision_abs_v_tent = args.pav1
        self.precision_p = args.pp

    def __str__(self):
        return 'ipcs1 - incremental pressure correction scheme with nonlinearity treated by Adam-Bashword + ' \
               'Crank-Nicolson and viscosity term treated semi-explicitly (Crank-Nicholson)'

    @staticmethod
    def setup_parser_options(parser):
        gs.GeneralSolver.setup_parser_options(parser)
        parser.add_argument('-s', '--solvers', help='Solvers', choices=['direct', 'krylov'], default='krylov')
        parser.add_argument('--prv1', help='relative tentative velocity Krylov solver precision', type=int, default=6)
        parser.add_argument('--pav1', help='absolute tentative velocity Krylov solver precision', type=int, default=10)
        parser.add_argument('--pp', help='pressure Krylov solver precision', type=int, default=10)
        parser.add_argument('-b', '--bc', help='Pressure boundary condition mode',
                            choices=['outflow', 'nullspace', 'nullspace_s', 'lagrange'], default='outflow')
        parser.add_argument('--precV', help='Preconditioner for tentative velocity solver', type=str, default='ilu')
        parser.add_argument('--precP', help='Preconditioner for pressure solver', choices=['hypre_amg', 'ilu'],
                            default='hypre_amg')
        parser.add_argument('-r', help='Use rotation scheme', action='store_true')
        parser.add_argument('-B', help='Use no BC in correction step', action='store_true')
        parser.add_argument('--fo', help='Force Neumann outflow boundary for pressure', action='store_true')
        parser.add_argument('--laplace', help='Use laplace(u) instead of div(symgrad(u))', action='store_true')
        parser.add_argument('--stab', help='Use stabilization (positive constant)', type=float, default=0.)
        parser.add_argument('--bcv', help='Oufflow BC for velocity (with stress formulation)',
                            choices=['CDN', 'DDN', 'LAP'], default='CDN')
        parser.add_argument('--ema', help='Use EMA conserving scheme for convection term', action='store_true')
        # described in Charnyi, Heister, Olshanskii, Rebholz:
        # "On conservation laws of Navier-Stokes Galerkin discretizations" (2016)
        parser.add_argument('--cs', help='Use consistent SUPG stabilisation.', action='store_true')
        parser.add_argument('--cbcDelta', help='Use simpler cbcflow parameter for SUPG', action='store_true')

    def solve(self, problem):
        self.problem = problem
        doSave = problem.doSave
        save_this_step = False
        onlyVel = problem.saveOnlyVel
        dt = self.metadata['dt']

        nu = Constant(self.problem.nu)
        # TODO check proper use of watches
        self.tc.init_watch('init', 'Initialization', True, count_to_percent=False)
        self.tc.init_watch('rhs', 'Assembled right hand side', True, count_to_percent=True)
        self.tc.init_watch('updateBC', 'Updated velocity BC', True, count_to_percent=True)
        self.tc.init_watch('applybc1', 'Applied velocity BC 1st step', True, count_to_percent=True)
        self.tc.init_watch('applybc3', 'Applied velocity BC 3rd step', True, count_to_percent=True)
        self.tc.init_watch('applybcP', 'Applied pressure BC or othogonalized rhs', True, count_to_percent=True)
        self.tc.init_watch('assembleMatrices', 'Initial matrix assembly', False, count_to_percent=True)
        self.tc.init_watch('solve 1', 'Running solver on 1st step', True, count_to_percent=True)
        self.tc.init_watch('solve 2', 'Running solver on 2nd step', True, count_to_percent=True)
        self.tc.init_watch('solve 3', 'Running solver on 3rd step', True, count_to_percent=True)
        self.tc.init_watch('solve 4', 'Running solver on 4th step', True, count_to_percent=True)
        self.tc.init_watch('assembleA1', 'Assembled A1 matrix (without stabiliz.)', True, count_to_percent=True)
        self.tc.init_watch('assembleA1stab', 'Assembled A1 stabilization', True, count_to_percent=True)
        self.tc.init_watch('next', 'Next step assignments', True, count_to_percent=True)
        self.tc.init_watch('saveVel', 'Saved velocity', True)

        self.tc.start('init')

        # Define function spaces (P2-P1)
        mesh = self.problem.mesh
        self.V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
        self.Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
        self.PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)
        self.D = FunctionSpace(mesh, "Lagrange", 1)   # velocity divergence space
        if self.bc == 'lagrange':
            L = FunctionSpace(mesh, "R", 0)
            QL = self.Q*L

        problem.initialize(self.V, self.Q, self.PS, self.D)

        # Define trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        if self.bc == 'lagrange':
            (pQL, rQL) = TrialFunction(QL)
            (qQL, lQL) = TestFunction(QL)
        else:
            p = TrialFunction(self.Q)
            q = TestFunction(self.Q)

        n = FacetNormal(mesh)
        I = Identity(u.geometric_dimension())

        # Initial conditions: u0 velocity at previous time step u1 velocity two time steps back p0 previous pressure
        [u1, u0, p0] = self.problem.get_initial_conditions([{'type': 'v', 'time': -dt},
                                                          {'type': 'v', 'time': 0.0},
                                                          {'type': 'p', 'time': 0.0}])

        if doSave:
            problem.save_vel(False, u0, 0.0)
            problem.save_vel(True, u0, 0.0)

        u_ = Function(self.V)         # current tentative velocity
        u_cor = Function(self.V)         # current corrected velocity
        if self.bc == 'lagrange':
            p_QL = Function(QL)    # current pressure or pressure help function from rotation scheme
            pQ = Function(self.Q)     # auxiliary function for conversion between QL.sub(0) and Q
        else:
            p_ = Function(self.Q)         # current pressure or pressure help function from rotation scheme
        p_mod = Function(self.Q)      # current modified pressure from rotation scheme

        # Define coefficients
        k = Constant(self.metadata['dt'])
        f = Constant((0, 0, 0))

        # Define forms
        # step 1: Tentative velocity, solve to u_
        u_ext = 1.5*u0 - 0.5*u1  # extrapolation for convection term

        # Stabilisation
        h = CellSize(mesh)
        # CBC delta:
        if self.cbcDelta:
            delta = Constant(self.stabCoef)*h/(sqrt(inner(u_ext, u_ext))+h)
        else:
            delta = Constant(self.stabCoef)*h**2/(2*nu*k + k*h*inner(u_ext, u_ext)+h**2)

        if self.use_full_SUPG:
            v1 = v + delta*0.5*k*dot(grad(v), u_ext)
            parameters['form_compiler']['quadrature_degree'] = 6
        else:
            v1 = v

        def nonlinearity(function):
            if self.use_ema:
               return 2*inner(dot(sym(grad(function)), u_ext), v1) * dx + inner(div(function)*u_ext, v1) * dx
                # return 2*inner(dot(sym(grad(function)), u_ext), v) * dx + inner(div(u_ext)*function, v) * dx
                # QQ implement this way?
            else:
                return inner(dot(grad(function), u_ext), v1) * dx

        def diffusion(fce):
            if self.useLaplace:
                return nu*inner(grad(fce), grad(v1)) * dx
            else:
                form = inner(nu * 2 * sym(grad(fce)), sym(grad(v1))) * dx
                if self.bcv == 'CDN':
                    # IMP will work only if p=0 on output, or we must add term
                    # inner(p0*n, v)*problem.get_outflow_measure_form() to avoid boundary layer
                    return form
                if self.bcv == 'LAP':
                    return form - inner(nu*dot(grad(fce).T, n), v1)  * problem.get_outflow_measure_form()
                if self.bcv == 'DDN':
                    # IMP will work only if p=0 on output, or we must add term
                    # inner(p0*n, v)*problem.get_outflow_measure_form() to avoid boundary layer
                    return form  # additional term must be added to non-constant part

        def pressure_rhs():
            if self.useLaplace or self.bcv == 'LAP':
                return inner(p0, div(v1)) * dx - inner(p0*n, v1) * problem.get_outflow_measure_form()
                # NT term inner(inner(p, n), v) is 0 when p=0 on outflow
            else:
                return inner(p0, div(v1)) * dx

        a1_const = (1./k)*inner(u, v1)*dx + diffusion(0.5*u)
        a1_change = nonlinearity(0.5*u)
        if self.bcv == 'DDN':
            # IMP Problem: Does not penalize influx for current step, only for the next one
            # IMP this can lead to oscilation: DDN correct next step, but then u_ext is OK so in next step DDN is not used, leading to new influx...
            # u and u_ext cannot be switched, min_value is nonlinear function
            a1_change += -0.5*min_value(Constant(0.), inner(u_ext, n))*inner(u, v1)*problem.get_outflow_measure_form()
            # IMP works only with uflacs compiler

        L1 = (1./k)*inner(u0, v1)*dx - nonlinearity(0.5*u0) - diffusion(0.5*u0) + pressure_rhs()
        if self.bcv == 'DDN':
            L1 += 0.5*min_value(0., inner(u_ext, n))*inner(u0, v1)*problem.get_outflow_measure_form()

        # Non-consistent SUPG stabilisation
        if self.stabilize and not self.use_full_SUPG:
            # a1_stab = delta*inner(dot(grad(u), u_ext), dot(grad(v), u_ext))*dx
            a1_stab = 0.5*delta*inner(dot(grad(u), u_ext), dot(grad(v), u_ext))*dx(None, {'quadrature_degree': 6})
            # NT optional: use Crank Nicolson in stabilisation term: change RHS
            # L1 += -0.5*delta*inner(dot(grad(u0), u_ext), dot(grad(v), u_ext))*dx(None, {'quadrature_degree': 6})

        outflow_area = Constant(problem.outflow_area)
        need_outflow = Constant(0.0)
        if self.useRotationScheme:
            # Rotation scheme
            if self.bc == 'lagrange':
                F2 = inner(grad(pQL), grad(qQL))*dx + (1./k)*qQL*div(u_)*dx + pQL*lQL*dx + qQL*rQL*dx
            else:
                F2 = inner(grad(p), grad(q))*dx + (1./k)*q*div(u_)*dx
        else:
            # Projection, solve to p_
            if self.bc == 'lagrange':
                F2 = inner(grad(pQL - p0), grad(qQL))*dx + (1./k)*qQL*div(u_)*dx + pQL*lQL*dx + qQL*rQL*dx
            else:
                if self.forceOutflow and problem.can_force_outflow:
                    info('Forcing outflow.')
                    F2 = inner(grad(p - p0), grad(q))*dx + (1./k)*q*div(u_)*dx
                    for m in problem.get_outflow_measures():
                        F2 += (1./k)*(1./outflow_area)*need_outflow*q*m
                else:
                    F2 = inner(grad(p - p0), grad(q))*dx + (1./k)*q*div(u_)*dx
        a2, L2 = system(F2)

        # step 3: Finalize, solve to u_
        if self.useRotationScheme:
            # Rotation scheme
            if self.bc == 'lagrange':
                F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_QL.sub(0)), v)*dx
            else:
                F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_), v)*dx
        else:
            if self.bc == 'lagrange':
                F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_QL.sub(0) - p0), v)*dx
            else:
                F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_ - p0), v)*dx
        a3, L3 = system(F3)

        if self.useRotationScheme:
            # Rotation scheme: modify pressure
            if self.bc == 'lagrange':
                pr = TrialFunction(self.Q)
                qr = TestFunction(self.Q)
                F4 = (pr - p0 - p_QL.sub(0) + nu*div(u_))*qr*dx
            else:
                F4 = (p - p0 - p_ + nu*div(u_))*q*dx
            # TODO zkusit, jestli to nebude rychlejsi? nepocitat soustavu, ale p.assign(...), nutno project(div(u),Q) coz je pocitani podobne soustavy
            # TODO zkusit v project zadat solver_type='lu' >> primy resic by mel byt efektivnejsi
            a4, L4 = system(F4)

        # Assemble matrices
        self.tc.start('assembleMatrices')
        A1_const = assemble(a1_const)  # need to be here, so A1 stays one Python object during repeated assembly
        A1_change = A1_const.copy()  # copy to get matrix with same sparse structure (data will be overwriten)
        if self.stabilize and not self.use_full_SUPG:
            A1_stab = A1_const.copy()  # copy to get matrix with same sparse structure (data will be overwriten)
        A2 = assemble(a2)
        A3 = assemble(a3)
        if self.useRotationScheme:
            A4 = assemble(a4)
        self.tc.end('assembleMatrices')

        if self.solvers == 'direct':
            self.solver_vel_tent = LUSolver('mumps')
            self.solver_vel_cor = LUSolver('mumps')
            self.solver_p = LUSolver('umfpack')
            if self.useRotationScheme:
                self.solver_rot = LUSolver('umfpack')
        else:
            # NT not needed, chosen not to use hypre_parasails
            # if self.prec_v == 'hypre_parasails':  # in FEniCS 1.6.0 inaccessible using KrylovSolver class
            #     self.solver_vel_tent = PETScKrylovSolver('gmres')   # PETSc4py object
            #     self.solver_vel_tent.ksp().getPC().setType('hypre')
            #     PETScOptions.set('pc_hypre_type', 'parasails')
            #     # this is global setting, but preconditioners for pressure solvers are set by their constructors
            # else:
            self.solver_vel_tent = KrylovSolver('gmres', self.prec_v)   # nonsymetric > gmres
            # IMP cannot use 'ilu' in parallel (choose different default option)
            self.solver_vel_cor = KrylovSolver('cg', 'hypre_amg')   # nonsymetric > gmres
            self.solver_p = KrylovSolver('cg', self.prec_p)          # symmetric > CG
            if self.useRotationScheme:
                self.solver_rot = KrylovSolver('cg', self.prec_p)

        solver_options = {'monitor_convergence': True, 'maximum_iterations': 1000, 'nonzero_initial_guess': True}
        # 'nonzero_initial_guess': True   with  solver.solbe(A, u, b) means that
        # Solver will use anything stored in u as an initial guess

        # Get the nullspace if there are no pressure boundary conditions
        foo = Function(self.Q)     # auxiliary vector for setting pressure nullspace
        if self.bc in ['nullspace', 'nullspace_s']:
            null_vec = Vector(foo.vector())
            self.Q.dofmap().set(null_vec, 1.0)
            null_vec *= 1.0/null_vec.norm('l2')
            self.null_space = VectorSpaceBasis([null_vec])
            if self.bc == 'nullspace':
                as_backend_type(A2).set_nullspace(self.null_space)

        # apply global options for Krylov solvers
        self.solver_vel_tent.parameters['relative_tolerance'] = 10 ** (-self.precision_rel_v_tent)
        self.solver_vel_tent.parameters['absolute_tolerance'] = 10 ** (-self.precision_abs_v_tent)
        self.solver_vel_cor.parameters['relative_tolerance'] = 10E-12
        self.solver_vel_cor.parameters['absolute_tolerance'] = 10E-4
        self.solver_p.parameters['relative_tolerance'] = 10**(-self.precision_p)
        self.solver_p.parameters['absolute_tolerance'] = 10E-10
        if self.useRotationScheme:
            self.solver_rot.parameters['relative_tolerance'] = 10**(-self.precision_p)
            self.solver_rot.parameters['absolute_tolerance'] = 10E-10

        if self.solvers == 'krylov':
            for solver in [self.solver_vel_tent, self.solver_vel_cor, self.solver_p, self.solver_rot] if \
                    self.useRotationScheme else [self.solver_vel_tent, self.solver_vel_cor, self.solver_p]:
                for key, value in solver_options.items():
                    try:
                        solver.parameters[key] = value
                    except KeyError:
                        info('Invalid option %s for KrylovSolver' % key)
                        return 1
                solver.parameters['preconditioner']['structure'] = 'same'
                # matrices A2-A4 do not change, so we can reuse preconditioners

        self.solver_vel_tent.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        # matrix A1 changes every time step, so change of preconditioner must be allowed

        if self.bc == 'lagrange':
            fa = FunctionAssigner(self.Q, QL.sub(0))

        # boundary conditions
        bcu, bcp = problem.get_boundary_conditions(self.bc == 'outflow', self.V, self.Q)
        self.tc.end('init')
        # Time-stepping
        info("Running of Incremental pressure correction scheme n. 1")
        ttime = self.metadata['time']
        t = dt
        step = 1
        while t < (ttime + dt/2.0):
            info("t = %f" % t)
            self.problem.update_time(t, step)
            if self.MPI_rank == 0:
                problem.write_status_file(t)

            if doSave:
                save_this_step = problem.save_this_step

            # DDN debug
            # u_ext_in = assemble(inner(u_ext, n)*problem.get_outflow_measure_form())
            # DDN_triggered = assemble(min_value(Constant(0.), inner(u_ext, n))*problem.get_outflow_measure_form())
            # print('DDN: u_ext*n dSout = ', u_ext_in)
            # print('DDN: negative part of u_ext*n dSout = ', DDN_triggered)

            # assemble matrix (it depends on solution)
            self.tc.start('assembleA1')
            assemble(a1_change, tensor=A1_change)  # assembling into existing matrix is faster than assembling new one
            A1 = A1_const.copy()  # we dont want to change A1_const
            A1.axpy(1, A1_change, True)
            self.tc.end('assembleA1')
            self.tc.start('assembleA1stab')
            if self.stabilize and not self.use_full_SUPG:
                assemble(a1_stab, tensor=A1_stab)  # assembling into existing matrix is faster than assembling new one
                A1.axpy(1, A1_stab, True)
            self.tc.end('assembleA1stab')

            # Compute tentative velocity step
            begin("Computing tentative velocity")
            self.tc.start('rhs')
            b = assemble(L1)
            self.tc.end('rhs')
            self.tc.start('applybc1')
            [bc.apply(A1, b) for bc in bcu]
            self.tc.end('applybc1')
            try:
                self.tc.start('solve 1')
                self.solver_vel_tent.solve(A1, u_.vector(), b)
                self.tc.end('solve 1')
                if save_this_step:
                    self.tc.start('saveVel')
                    problem.save_vel(True, u_, t)
                    self.tc.end('saveVel')
                if save_this_step and not onlyVel:
                    problem.save_div(True, u_)
                problem.compute_err(True, u_, t)
                problem.compute_div(True, u_)
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            end()

            # DDN debug
            # u_ext_in = assemble(inner(u_, n)*problem.get_outflow_measure_form())
            # DDN_triggered = assemble(min_value(Constant(0.), inner(u_, n))*problem.get_outflow_measure_form())
            # print('DDN: u_tent*n dSout = ', u_ext_in)
            # print('DDN: negative part of u_tent*n dSout = ', DDN_triggered)

            if self.useRotationScheme:
                begin("Computing tentative pressure")
            else:
                begin("Computing pressure")
            if self.forceOutflow and problem.can_force_outflow:
                out = problem.compute_outflow(u_)
                info('Tentative outflow: %f' % out)
                n_o = -problem.last_inflow-out
                info('Needed outflow: %f' % n_o)
                need_outflow.assign(n_o)
            self.tc.start('rhs')
            b = assemble(L2)
            self.tc.end('rhs')
            self.tc.start('applybcP')
            [bc.apply(A2, b) for bc in bcp]
            if self.bc in ['nullspace', 'nullspace_s']:
                self.null_space.orthogonalize(b)
            self.tc.end('applybcP')
            try:
                self.tc.start('solve 2')
                if self.bc == 'lagrange':
                    self.solver_p.solve(A2, p_QL.vector(), b)
                else:
                    self.solver_p.solve(A2, p_.vector(), b)
                self.tc.end('solve 2')
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            if self.useRotationScheme:
                foo = Function(self.Q)
                if self.bc == 'lagrange':
                    fa.assign(pQ, p_QL.sub(0))
                    foo.assign(pQ + p0)
                else:
                    foo.assign(p_+p0)
                problem.averaging_pressure(foo)
                if save_this_step and not onlyVel:
                    problem.save_pressure(True, foo)
            else:
                if self.bc == 'lagrange':
                    fa.assign(pQ, p_QL.sub(0))
                    problem.averaging_pressure(pQ)
                    if save_this_step and not onlyVel:
                        problem.save_pressure(False, pQ)
                else:
                    # we do not want to change p=0 on outflow, it conflicts with do-nothing conditions
                    foo = Function(self.Q)
                    foo.assign(p_)
                    problem.averaging_pressure(foo)
                    if save_this_step and not onlyVel:
                        problem.save_pressure(False, foo)
            end()

            begin("Computing corrected velocity")
            self.tc.start('rhs')
            b = assemble(L3)
            self.tc.end('rhs')
            if not self.B:
                self.tc.start('applybc3')
                [bc.apply(A3, b) for bc in bcu]
                self.tc.end('applybc3')
            try:
                self.tc.start('solve 3')
                self.solver_vel_cor.solve(A3, u_cor.vector(), b)
                self.tc.end('solve 3')
                problem.compute_err(False, u_cor, t)
                problem.compute_div(False, u_cor)
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            if save_this_step:
                self.tc.start('saveVel')
                problem.save_vel(False, u_cor, t)
                self.tc.end('saveVel')
            if save_this_step and not onlyVel:
                problem.save_div(False, u_cor)
            end()

            # DDN debug
            # u_ext_in = assemble(inner(u_cor, n)*problem.get_outflow_measure_form())
            # DDN_triggered = assemble(min_value(Constant(0.), inner(u_cor, n))*problem.get_outflow_measure_form())
            # print('DDN: u_cor*n dSout = ', u_ext_in)
            # print('DDN: negative part of u_cor*n dSout = ', DDN_triggered)

            if self.useRotationScheme:
                begin("Rotation scheme pressure correction")
                self.tc.start('rhs')
                b = assemble(L4)
                self.tc.end('rhs')
                try:
                    self.tc.start('solve 4')
                    self.solver_rot.solve(A4, p_mod.vector(), b)
                    self.tc.end('solve 4')
                except RuntimeError as inst:
                    problem.report_fail(t)
                    return 1
                problem.averaging_pressure(p_mod)
                if save_this_step and not onlyVel:
                    problem.save_pressure(False, p_mod)
                end()

            # compute functionals (e. g. forces)
            problem.compute_functionals(u_cor,
                                        p_mod if self.useRotationScheme else (pQ if self.bc == 'lagrange' else p_), t)

            # Move to next time step
            self.tc.start('next')
            u1.assign(u0)
            u0.assign(u_cor)
            u_.assign(u_cor)  # use corretced velocity as initial guess in first step

            if self.useRotationScheme:
                p0.assign(p_mod)
            else:
                if self.bc == 'lagrange':
                    p0.assign(pQ)
                else:
                    p0.assign(p_)

            t = round(t + dt, 6)  # round time step to 0.000001
            step += 1
            self.tc.end('next')

        info("Finished: Incremental pressure correction scheme n. 1")
        problem.report()
        return 0
