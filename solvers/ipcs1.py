from __future__ import print_function
from dolfin import Function, VectorFunctionSpace, FunctionSpace, assemble
from dolfin.cpp.common import info, begin, end
from dolfin.cpp.function import FunctionAssigner
from dolfin.cpp.la import LUSolver, KrylovSolver, as_backend_type, VectorSpaceBasis, Vector
from dolfin.functions import TrialFunction, TestFunction, Constant
from ufl import dot, dx, grad, system, div, inner

import general_solver as gs

# QQ split rotation, lagrange scheme?
# (implement as this Solver subclass? Can class be subclass of class with same name?)
# in current state almost no code is saved by subclassing this, one must rewrite whole solve()


class Solver(gs.GeneralSolver):
    def __init__(self, args, tc, metadata):
        gs.GeneralSolver.__init__(self, args, tc, metadata)
        self.metadata['hasTentativeV'] = True

        self.solver_vel = None
        self.solver_p = None
        self.solver_rot = None
        self.nullspace = None

        # input parameters
        self.bc = args.bc
        self.solvers = args.solvers
        self.useRotationScheme = args.r
        self.metadata['hasTentativeP'] = self.useRotationScheme

        self.B = args.B
        self.prec = args.prec
        self.precision = args.precision

    def __str__(self):
        return 'ipcs1 - incremental pressure correction scheme with nonlinearity treated by Adam-Bashword + ' \
               'Crank-Nicolson and viscosity term treated semi-explicitly (Crank-Nicholson)'

    @staticmethod
    def setup_parser_options(parser):
        gs.GeneralSolver.setup_parser_options(parser)
        parser.add_argument('-s', '--solvers', help='Solvers', choices=['direct', 'krylov'], default='krylov')
        parser.add_argument('-p', '--precision', help='Krylov solver precision', type=int, default=6)
        parser.add_argument('-b', '--bc', help='Pressure boundary condition mode',
                            choices=['outflow', 'nullspace', 'nullspace_s', 'lagrange'], default='outflow')
        parser.add_argument('--prec', help='Preconditioner for pressure solver', choices=['hypre_amg', 'ilu'],
                            default='hypre_amg')
        parser.add_argument('-r', help='Use rotation scheme', action='store_true')
        parser.add_argument('-B', help='Use no BC in correction step', action='store_true')

    def solve(self, problem):
        self.problem = problem
        doSave = problem.doSave

        self.tc.start('init')
        nu = Constant(self.problem.nu)
        # TODO check proper use of watches
        self.tc.init_watch('init', 'Initialization', True)
        self.tc.init_watch('rhs', 'Assembled right hand side', True)
        self.tc.init_watch('updateBC', 'Updated velocity BC', True)
        self.tc.init_watch('applybc1', 'Applied velocity BC 1st step', True)
        self.tc.init_watch('applybc3', 'Applied velocity BC 3rd step', True)
        self.tc.init_watch('applybcP', 'Applied pressure BC or othogonalized rhs', True)
        self.tc.init_watch('assembleMatrices', 'Initial matrix assembly', False)
        self.tc.init_watch('solve 1', 'Running solver on 1st step', True)
        self.tc.init_watch('solve 2', 'Running solver on 2nd step', True)
        self.tc.init_watch('solve 3', 'Running solver on 3rd step', True)
        self.tc.init_watch('solve 4', 'Running solver on 4th step', True)
        self.tc.init_watch('assembleA1', 'Assembled A1 matrix', True)
        self.tc.init_watch('computePG', 'Computed pressure gradient', True)
        self.tc.init_watch('next', 'Next step assignments', True)

        # Define function spaces (P2-P1)
        mesh = self.problem.mesh
        self.V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
        self.Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
        self.PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)
        if self.bc == 'lagrange':
            L = FunctionSpace(mesh, "R", 0)
            QL = self.Q*L

        # Define trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        if self.bc == 'lagrange':
            (pQL, rQL) = TrialFunction(QL)
            (qQL, lQL) = TestFunction(QL)
        else:
            p = TrialFunction(self.Q)
            q = TestFunction(self.Q)

        # Initial conditions: u0 velocity at previous time step u1 velocity two time steps back p0 previous pressure
        u0, p0 = self.problem.get_initial_conditions(self.V, self.Q)
        u1 = u0

        # if doSave:
        #     rm.save_vel(False, u0, 0.0)
        #     rm.save_vel(True, u0, 0.0)

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
        U = 0.5*(u + u0)
        U_ = 1.5*u0 - 0.5*u1

        nonlinearity = inner(dot(grad(U), U_), v)*dx

        F1 = (1./k)*inner(u - u0, v)*dx + nonlinearity\
            + nu*inner(grad(U), grad(v))*dx + inner(grad(p0), v)*dx\
            - inner(f, v)*dx     # solve to u_
        a1, L1 = system(F1)

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
        A1 = assemble(a1)  # need to be here, so A1 stays one Python object during repeated assembly
        A2 = assemble(a2)
        A3 = assemble(a3)
        if self.useRotationScheme:
            A4 = assemble(a4)
        self.tc.end('assembleMatrices')

        if self.solvers == 'direct':
            self.solver_vel = LUSolver('mumps')
            self.solver_p = LUSolver('umfpack')
            if self.useRotationScheme:
                self.solver_rot = LUSolver('umfpack')
        else:
            self.solver_vel = KrylovSolver('gmres', 'ilu')   # nonsymetric > gmres  # IFNEED try hypre_amg
            self.solver_p = KrylovSolver('cg', self.prec)          # symmetric > CG
            if self.useRotationScheme:
                self.solver_rot = KrylovSolver('cg', self.prec)

        solver_options = {'absolute_tolerance': 10E-12, 'relative_tolerance': 10**(-self.precision),
                          'monitor_convergence': True, 'maximum_iterations': 500}

        # Get the nullspace if there are no pressure boundary conditions
        foo = Function(self.Q)     # auxiliary vector for setting pressure nullspace
        if self.bc in ['nullspace', 'nullspace_s']:
            null_vec = Vector(foo.vector())
            self.Q.dofmap().set(null_vec, 1.0)
            null_vec *= 1.0/null_vec.norm('l2')
            null_space = VectorSpaceBasis([null_vec])
            if self.bc == 'nullspace':
                as_backend_type(A2).set_nullspace(null_space)

        # apply global options for Krylov solvers
        if self.solvers == 'krylov':
            for solver in [self.solver_vel, self.solver_p, self.solver_rot] if self.useRotationScheme else \
                    [self.solver_vel, self.solver_p]:
                for key, value in solver_options.items():
                    try:
                        solver.parameters[key] = value
                    except KeyError:
                        print('Invalid option %s for KrylovSolver' % key)
                        exit()
                solver.parameters['preconditioner']['structure'] = 'same'

        if self.bc == 'lagrange':
            fa = FunctionAssigner(self.Q, QL.sub(0))

        problem.initialize(self.V, self.Q, self.PS)

        # boundary conditions
        bcu, bcp = problem.get_boundary_conditions(self.V, self.Q, self.bc == 'outflow')
        self.tc.end('init')
        # Time-stepping
        info("Running of Incremental pressure correction scheme n. 1")
        dt = self.metadata['dt']
        ttime = self.metadata['time']
        t = dt
        while t < (ttime + dt/2.0):
            print("t = ", t)
            self.problem.update_time(t)
            self.write_status_file(t, problem.last_status_functional, problem.status_functional_str)

            # assemble matrix (it depends on solution)
            self.tc.start('assembleA1')
            assemble(a1, tensor=A1)  # tensor must by of type GenericMatrix before using this assemble
            self.tc.end('assembleA1')

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
                self.solver_vel.solve(A1, u_.vector(), b)
                self.tc.end('solve 1')
            except RuntimeError as inst:
                self.report_fail(t)
                exit()
            # rm.compute_err(True, u_, t)
            # rm.compute_div(True, u_)
            if doSave:
                self.tc.start('saveVel')
                problem.save_vel(True, u_, t)
                self.tc.end('saveVel')
            #     rm.save_div(True, u_)
            end()

            if self.useRotationScheme:
                begin("Computing tentative pressure")
            else:
                begin("Computing pressure")
            self.tc.start('rhs')
            b = assemble(L2)
            self.tc.end('rhs')
            self.tc.start('applybcP')
            [bc.apply(A2, b) for bc in bcp]
            if self.bc in ['nullspace', 'nullspace_s']:
                self.nullspace.orthogonalize(b)
            self.tc.end('applybcP')
            # print(A2, b, p_.vector())
            try:
                self.tc.start('solve 2')
                if self.bc == 'lagrange':
                    self.solver_p.solve(A2, p_QL.vector(), b)
                else:
                    self.solver_p.solve(A2, p_.vector(), b)
                self.tc.end('solve 2')
            except RuntimeError as inst:
                self.report_fail(t)
                exit()
            self.tc.start('saveP')
            if self.useRotationScheme:
                foo = Function(self.Q)
                if self.bc == 'lagrange':
                    fa.assign(pQ, p_QL.sub(0))
                    foo.assign(pQ + p0)
                else:
                    foo.assign(p_+p0)
                problem.averaging_pressure(foo)
                problem.save_pressure(True, foo)
            else:
                if self.bc == 'lagrange':
                    fa.assign(pQ, p_QL.sub(0))
                    problem.averaging_pressure(pQ)
                    problem.save_pressure(False, pQ)
                else:
                    problem.averaging_pressure(p_)
                    problem.save_pressure(False, p_)
            self.tc.end('saveP')
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
                self.solver_vel.solve(A3, u_cor.vector(), b)
                self.tc.end('solve 3')
            except RuntimeError as inst:
                self.report_fail(t)
                exit()
            # rm.compute_err(False, u_cor, t)
            # rm.compute_div(False, u_cor)
            # if doSave:
            #     rm.save_vel(False, u_cor, t)
            #     rm.save_div(False, u_cor)
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
                    self.report_fail(t)
                    exit()
                self.tc.start('saveP')
                problem.averaging_pressure(p_mod)
                problem.save_pressure(False, p_mod)
                self.tc.end('saveP')
                end()

            # plot(p_, title='tent')
            # plot(p_mod, title='cor', interactive=True)
            # exit()

            # compute force on wall
            # rm.compute_force(u_cor, p_mod if useRotationScheme else (pQ if args.bc == 'lagrange' else p_), n, t)  # NT clean somehow?

            # Move to next time step
            self.tc.start('next')
            u1.assign(u0)
            u0.assign(u_cor)
            if self.useRotationScheme:
                p0.assign(p_mod)
            else:
                if self.bc == 'lagrange':
                    p0.assign(pQ)
                else:
                    p0.assign(p_)

            t = round(t + dt, 4)  # round time step to 0.0001
            self.tc.end('next')

        info("Finished: Incremental pressure correction scheme n. 1")
        problem.report()

