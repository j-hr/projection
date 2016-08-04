from __future__ import print_function

from dolfin import Function, VectorFunctionSpace, FunctionSpace, assemble, CellSize, DOLFIN_EPS, parameters, \
    project
from dolfin.cpp.common import info, begin, end
from dolfin.cpp.la import LUSolver, KrylovSolver, as_backend_type, VectorSpaceBasis, Vector
from dolfin.functions import TrialFunction, TestFunction, Constant, FacetNormal
from ufl import dot, dx, grad, system, div, inner, sym, Identity, sqrt, min_value
import general_solver as gs


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

    def __str__(self):
        return 'ipcs1 - incremental pressure correction scheme with nonlinearity treated by Adam-Bashword + ' \
               'Crank-Nicolson and viscosity term treated semi-explicitly (Crank-Nicholson)'

    @staticmethod
    def setup_parser_options(parser):
        gs.GeneralSolver.setup_parser_options(parser)
        parser.add_argument('--stab', help='Use stabilization (positive constant)', type=float, default=0.)
        parser.add_argument('--cs', help='Use consistent SUPG stabilisation.', action='store_true')
        parser.add_argument('--cbc_tau', help='Use simpler cbcflow parameter for SUPG', action='store_true')
        parser.add_argument('-r', help='Use rotation scheme', action='store_true')
        parser.add_argument('-B', help='Use no BC in correction step', action='store_true')
        parser.add_argument('-s', '--solvers', help='Solvers', choices=['direct', 'krylov'], default='krylov')
        parser.add_argument('-b', '--bc', help='Pressure boundary condition mode',
                            choices=['outflow', 'nullspace'], default='outflow')
        # 'outflow' should be used with direct solvers (cannot LU-decompose singular matrix)
        parser.add_argument('--prv1', help='relative tentative velocity Krylov solver precision', type=int, default=6)
        parser.add_argument('--pav1', help='absolute tentative velocity Krylov solver precision', type=int, default=10)
        parser.add_argument('--pap', help='pressure Krylov solver absolute precision', type=int, default=6)
        parser.add_argument('--prp', help='pressure Krylov solver relative precision', type=int, default=10)
        parser.add_argument('--precV', help='Preconditioner for tentative velocity GMRES solver', type=str, default='sor')
        parser.add_argument('--precVC', help='Preconditioner for corrected velocity CG solver', type=str, default='sor')
        parser.add_argument('--precP', help='Preconditioner for 2nd step solver (Poisson)',
                            choices=['hypre_amg', 'ilu', 'sor'], default='sor')
        parser.add_argument('--solP', help='2nd step solver (Poisson)', type=str, default='cg')
        # choices=['cg', 'gmres', 'richardson', 'tfqmr', ...]
        parser.add_argument('--Prestart', help='2nd step solver GMRES restart', type=int, default=-1)
        parser.add_argument('--Vrestart', help='1st step solver GMRES restart', type=int, default=-1)
        parser.add_argument('--fo', help='Force Neumann outflow boundary for pressure', action='store_true')
        parser.add_argument('--laplace', help='Use laplace(u) instead of div(symgrad(u))', action='store_true')
        parser.add_argument('--bcv', help='Outflow BC for velocity (with stress formulation)',
                            choices=['CDN', 'DDN', 'LAP'], default='CDN')
        parser.add_argument('--ema', help='Use EMA conserving scheme for convection term', action='store_true')
        # described in Charnyi, Heister, Olshanskii, Rebholz:
        # "On conservation laws of Navier-Stokes Galerkin discretizations" (2016)
        # it seems that it does not work with projection methods (probably because of dropping div u = 0 constraint)
        # not documented in readme

    def solve(self, problem):
        self.problem = problem
        doSave = problem.doSave
        save_this_step = False
        onlyVel = problem.saveOnlyVel
        dt = self.metadata['dt']

        nu = Constant(self.problem.nu)
        self.tc.init_watch('init', 'Initialization', True, count_to_percent=False)
        self.tc.init_watch('rhs', 'Assembled right hand side', True, count_to_percent=True)
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

        problem.initialize(self.V, self.Q, self.PS, self.D)

        # Define trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)

        n = FacetNormal(mesh)
        I = Identity(u.geometric_dimension())

        # Initial conditions: u0 velocity at previous time step u1 velocity two time steps back p0 previous pressure
        [u1, u0, p0] = self.problem.get_initial_conditions([{'type': 'v', 'time': -dt},
                                                          {'type': 'v', 'time': 0.0},
                                                          {'type': 'p', 'time': 0.0}])

        u_ = Function(self.V)         # current tentative velocity
        u_cor = Function(self.V)         # current corrected velocity
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
        if self.args.cbc_tau:
            # used in Simula cbcflow project
            tau = Constant(self.stabCoef)*h/(sqrt(inner(u_ext, u_ext))+h)
        else:
            # proposed in R. Codina: On stabilized finite element methods for linear systems of
            # convection-diffusion-reaction equations.
            tau = Constant(self.stabCoef)*k*h**2/(2*nu*k + k*h*sqrt(DOLFIN_EPS + inner(u_ext, u_ext))+h**2)
            # DOLFIN_EPS is added because of FEniCS bug that inner(u_ext, u_ext) can be negative when u_ext = 0

        if self.use_full_SUPG:
            v1 = v + tau*0.5*dot(grad(v), u_ext)
            parameters['form_compiler']['quadrature_degree'] = 6
        else:
            v1 = v

        def nonlinearity(function):
            if self.args.ema:
                return 2*inner(dot(sym(grad(function)), u_ext), v1) * dx + inner(div(function)*u_ext, v1) * dx
            else:
                return inner(dot(grad(function), u_ext), v1) * dx

        def diffusion(fce):
            if self.useLaplace:
                return nu*inner(grad(fce), grad(v1)) * dx
            else:
                form = inner(nu * 2 * sym(grad(fce)), sym(grad(v1))) * dx
                if self.bcv == 'CDN':
                    return form
                if self.bcv == 'LAP':
                    return form - inner(nu*dot(grad(fce).T, n), v1) * problem.get_outflow_measure_form()
                if self.bcv == 'DDN':
                    return form  # additional term must be added to non-constant part

        def pressure_rhs():
            return inner(p0, div(v1)) * dx - inner(p0*n, v1) * problem.get_outflow_measure_form()

        a1_const = (1./k)*inner(u, v1)*dx + diffusion(0.5*u)
        a1_change = nonlinearity(0.5*u)
        if self.bcv == 'DDN':
            # does not penalize influx for current step, only for the next one
            # this can lead to oscilation:
            # DDN correct next step, but then u_ext is OK so in next step DDN is not used, leading to new influx...
            # u and u_ext cannot be switched, min_value is nonlinear function
            a1_change += -0.5*min_value(Constant(0.), inner(u_ext, n))*inner(u, v1)*problem.get_outflow_measure_form()
            # NT works only with uflacs compiler

        L1 = (1./k)*inner(u0, v1)*dx - nonlinearity(0.5*u0) - diffusion(0.5*u0) + pressure_rhs()
        if self.bcv == 'DDN':
            L1 += 0.5*min_value(0., inner(u_ext, n))*inner(u0, v1)*problem.get_outflow_measure_form()

        # Non-consistent SUPG stabilisation
        if self.stabilize and not self.use_full_SUPG:
            # a1_stab = tau*inner(dot(grad(u), u_ext), dot(grad(v), u_ext))*dx
            a1_stab = 0.5*tau*inner(dot(grad(u), u_ext), dot(grad(v), u_ext))*dx(None, {'quadrature_degree': 6})
            # optional: to use Crank Nicolson in stabilisation term following change of RHS is needed:
            # L1 += -0.5*tau*inner(dot(grad(u0), u_ext), dot(grad(v), u_ext))*dx(None, {'quadrature_degree': 6})

        outflow_area = Constant(problem.outflow_area)
        need_outflow = Constant(0.0)
        if self.useRotationScheme:
            # Rotation scheme
            F2 = inner(grad(p), grad(q))*dx + (1./k)*q*div(u_)*dx
        else:
            # Projection, solve to p_
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
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_), v)*dx
        else:
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_ - p0), v)*dx
        a3, L3 = system(F3)

        if self.useRotationScheme:
            # Rotation scheme: modify pressure
            F4 = (p - p0 - p_ + nu*div(u_))*q*dx
            a4, L4 = system(F4)

        # Assemble matrices
        self.tc.start('assembleMatrices')
        A1_const = assemble(a1_const)  # must be here, so A1 stays one Python object during repeated assembly
        A1_change = A1_const.copy()  # copy to get matrix with same sparse structure (data will be overwritten)
        if self.stabilize and not self.use_full_SUPG:
            A1_stab = A1_const.copy()  # copy to get matrix with same sparse structure (data will be overwritten)
        A2 = assemble(a2)
        A3 = assemble(a3)
        if self.useRotationScheme:
            A4 = assemble(a4)
        self.tc.end('assembleMatrices')

        if self.solvers == 'direct':
            self.solver_vel_tent = LUSolver('mumps')
            self.solver_vel_cor = LUSolver('mumps')
            self.solver_p = LUSolver('mumps')
            if self.useRotationScheme:
                self.solver_rot = LUSolver('mumps')
        else:
            # not needed, chosen not to use hypre_parasails:
            # if self.prec_v == 'hypre_parasails':  # in FEniCS 1.6.0 inaccessible using KrylovSolver class
            #     self.solver_vel_tent = PETScKrylovSolver('gmres')   # PETSc4py object
            #     self.solver_vel_tent.ksp().getPC().setType('hypre')
            #     PETScOptions.set('pc_hypre_type', 'parasails')
            #     # this is global setting, but preconditioners for pressure solvers are set by their constructors
            # else:
            self.solver_vel_tent = KrylovSolver('gmres', self.args.precV)   # nonsymetric > gmres
            # cannot use 'ilu' in parallel
            self.solver_vel_cor = KrylovSolver('cg', self.args.precVC)
            self.solver_p = KrylovSolver(self.args.solP, self.args.precP)    # almost (up to BC) symmetric > CG
            if self.useRotationScheme:
                self.solver_rot = KrylovSolver('cg', 'hypre_amg')

        # setup Krylov solvers
        if self.solvers == 'krylov':
            # Get the nullspace if there are no pressure boundary conditions
            foo = Function(self.Q)     # auxiliary vector for setting pressure nullspace
            if self.args.bc == 'nullspace':
                null_vec = Vector(foo.vector())
                self.Q.dofmap().set(null_vec, 1.0)
                null_vec *= 1.0/null_vec.norm('l2')
                self.null_space = VectorSpaceBasis([null_vec])
                as_backend_type(A2).set_nullspace(self.null_space)

            # apply global options for Krylov solvers
            solver_options = {'monitor_convergence': True, 'maximum_iterations': 1000, 'nonzero_initial_guess': True}
            # 'nonzero_initial_guess': True   with  solver.solve(A, u, b) means that
            # Solver will use anything stored in u as an initial guess
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

            self.solver_vel_tent.parameters['relative_tolerance'] = 10 ** (-self.args.prv1)
            self.solver_vel_tent.parameters['absolute_tolerance'] = 10 ** (-self.args.pav1)
            self.solver_vel_cor.parameters['relative_tolerance'] = 10E-12
            self.solver_vel_cor.parameters['absolute_tolerance'] = 10E-4
            self.solver_p.parameters['relative_tolerance'] = 10**(-self.args.prp)
            self.solver_p.parameters['absolute_tolerance'] = 10**(-self.args.pap)
            if self.useRotationScheme:
                self.solver_rot.parameters['relative_tolerance'] = 10E-10
                self.solver_rot.parameters['absolute_tolerance'] = 10E-10

            if self.args.Vrestart > 0:
                self.solver_vel_tent.parameters['gmres']['restart'] = self.args.Vrestart

            if self.args.solP == 'gmres' and self.args.Prestart > 0:
                self.solver_p.parameters['gmres']['restart'] = self.args.Prestart

        # boundary conditions
        bcu, bcp = problem.get_boundary_conditions(self.args.bc == 'outflow', self.V, self.Q)
        self.tc.end('init')
        # Time-stepping
        info("Running of Incremental pressure correction scheme n. 1")
        ttime = self.metadata['time']
        t = dt
        step = 1

        # debug function
        if problem.args.debug_rot:
            plot_cor_v = Function(self.V)

        while t < (ttime + dt/2.0):
            self.problem.update_time(t, step)
            if self.MPI_rank == 0:
                problem.write_status_file(t)

            if doSave:
                save_this_step = problem.save_this_step

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
                    problem.save_vel(True, u_)
                    self.tc.end('saveVel')
                if save_this_step and not onlyVel:
                    problem.save_div(True, u_)
                problem.compute_err(True, u_, t)
                problem.compute_div(True, u_)
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            end()

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
            if self.args.bc == 'nullspace':
                self.null_space.orthogonalize(b)
            self.tc.end('applybcP')
            try:
                self.tc.start('solve 2')
                self.solver_p.solve(A2, p_.vector(), b)
                self.tc.end('solve 2')
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            if self.useRotationScheme:
                foo = Function(self.Q)
                foo.assign(p_+p0)
                if save_this_step and not onlyVel:
                    problem.averaging_pressure(foo)
                    problem.save_pressure(True, foo)
            else:
                foo = Function(self.Q)
                foo.assign(p_)  # we do not want to change p_ by averaging
                if save_this_step and not onlyVel:
                    problem.averaging_pressure(foo)
                    problem.save_pressure(False, foo)
            end()

            begin("Computing corrected velocity")
            self.tc.start('rhs')
            b = assemble(L3)
            self.tc.end('rhs')
            if not self.args.B:
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
                problem.save_vel(False, u_cor)
                self.tc.end('saveVel')
            if save_this_step and not onlyVel:
                problem.save_div(False, u_cor)
            end()

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
                if save_this_step and not onlyVel:
                    problem.averaging_pressure(p_mod)
                    problem.save_pressure(False, p_mod)
                end()

                if problem.args.debug_rot:
                    # save applied pressure correction (expressed as a term added to RHS of next tentative vel. step)
                    # see comment next to argument definition
                    plot_cor_v.assign(project(k*grad(nu*div(u_)), self.V))
                    problem.fileDict['grad_cor']['file'] << (plot_cor_v, t)

            # compute functionals (e. g. forces)
            problem.compute_functionals(u_cor, p_mod if self.useRotationScheme else p_, t, step)

            # Move to next time step
            self.tc.start('next')
            u1.assign(u0)
            u0.assign(u_cor)
            u_.assign(u_cor)  # use corrected velocity as initial guess in first step

            if self.useRotationScheme:
                p0.assign(p_mod)
            else:
                p0.assign(p_)

            t = round(t + dt, 6)  # round time step to 0.000001
            step += 1
            self.tc.end('next')

        info("Finished: Incremental pressure correction scheme n. 1")
        problem.report()
        return 0
