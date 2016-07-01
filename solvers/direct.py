from __future__ import print_function
from dolfin import Function, VectorFunctionSpace, FunctionSpace, assemble, Expression, CellSize, DOLFIN_EPS, parameters, \
    plot, MixedFunctionSpace, DirichletBC, NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin.cpp.common import info, begin, end
from dolfin.cpp.function import FunctionAssigner
from dolfin.cpp.la import LUSolver, KrylovSolver, as_backend_type, VectorSpaceBasis, Vector, PETScKrylovSolver, \
    PETScOptions
from dolfin.functions import TrialFunction, TestFunction, Constant, FacetNormal, TestFunctions
from ufl import dot, dx, grad, system, div, inner, sym, Identity, transpose, nabla_grad, sqrt, min_value, split, \
    derivative

import general_solver as gs


class Solver(gs.GeneralSolver):
    def __init__(self, args, tc, metadata):
        gs.GeneralSolver.__init__(self, args, tc, metadata)
        self.metadata['hasTentativeV'] = False

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
        self.tc.init_watch('init', 'Initialization', True, count_to_percent=False)
        self.tc.init_watch('solve', 'Running nonlinear solver', True, count_to_percent=True)
        self.tc.init_watch('next', 'Next step assignments', True, count_to_percent=True)
        self.tc.init_watch('saveVel', 'Saved velocity', True)

        self.tc.start('init')

        mesh = self.problem.mesh

        # Define function spaces (P2-P1)
        self.V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
        self.Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
        self.W = MixedFunctionSpace([self.V, self.Q])
        self.PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)
        self.D = FunctionSpace(mesh, "Lagrange", 1)   # velocity divergence space

        # to assign solution in space W.sub(0) to Function(V) we need FunctionAssigner (cannot be assigned directly)
        fa = FunctionAssigner(self.V, self.W.sub(0))
        velSp = Function(self.V)

        problem.initialize(self.V, self.Q, self.PS, self.D)

        # Define unknown and test function(s) NS
        v, q = TestFunctions(self.W)
        w = Function(self.W)
        dw = TrialFunction(self.W)
        u, p = split(w)

        # Define fields
        n = FacetNormal(mesh)
        I = Identity(u.geometric_dimension())
        theta = 0.5  # Crank-Nicholson
        k = Constant(self.metadata['dt'])

        # Initial conditions: u0 velocity at previous time step u1 velocity two time steps back p0 previous pressure
        [u0, p0] = self.problem.get_initial_conditions([{'type': 'v', 'time': 0.0}, {'type': 'p', 'time': 0.0}])

        if doSave:
            problem.save_vel(False, u0, 0.0)

        # boundary conditions
        bcu, bcp = problem.get_boundary_conditions(self.bc == 'outflow', self.W.sub(0), self.W.sub(1))
        # NT bcp is not used

        # Define steady part of the equation
        def T(u):
            return -p * I + 2.0 * nu * sym(grad(u))

        def F(u, v, q):
            return (inner(T(u), grad(v)) - q * div(u)) * dx + inner(grad(u) * u, v) * dx

        # Define variational forms
        F_ns = (inner((u - u0), v) / k) * dx + (1.0 - theta) * F(u0, v, q) + theta * F(u, v, q)
        J_ns = derivative(F_ns, w, dw)
        # J_ns = derivative(F_ns, w)  # did not work

        # NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns, form_compiler_parameters=ffc_options)
        NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns)
        # (var. formulation, unknown, Dir. BC, jacobian, optional)
        NS_solver = NonlinearVariationalSolver(NS_problem)

        prm = NS_solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-08
        prm['newton_solver']['relative_tolerance'] = 1E-08
        # prm['newton_solver']['maximum_iterations'] = 45
        # prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = 'mumps'

        info(NS_solver.parameters, True)

        self.tc.end('init')

        # Time-stepping
        info("Running of direct method")
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

            # Compute
            begin("Solving NS ....")
            try:
                self.tc.start('solve')
                NS_solver.solve()
                self.tc.end('solve')
            except RuntimeError as inst:
                problem.report_fail(t)
                return 1
            end()

            # Extract solutions:
            (u, p) = w.split()
            fa.assign(velSp, u)
            # we are assigning twice (now and inside save_vel), but it works with one method save_vel for direct and
            #   projection (we could split save_vel to save one assign)

            if save_this_step:
                self.tc.start('saveVel')
                problem.save_vel(False, velSp, t)
                self.tc.end('saveVel')
            if save_this_step and not onlyVel:
                problem.save_div(False, u)
            problem.compute_err(False, u, t)
            problem.compute_div(False, u)

            # foo = Function(self.Q)
            # foo.assign(p)
            # problem.averaging_pressure(foo)
            # if save_this_step and not onlyVel:
            #     problem.save_pressure(False, foo)

            if save_this_step and not onlyVel:
                problem.save_pressure(False, p)

            # compute functionals (e. g. forces)
            problem.compute_functionals(u, p, t)

            # Move to next time step
            self.tc.start('next')
            u0.assign(velSp)
            t = round(t + dt, 6)  # round time step to 0.000001
            step += 1
            self.tc.end('next')

        info("Finished: direct method")
        problem.report()
        return 0
