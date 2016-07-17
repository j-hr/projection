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
        self.metadata['hasTentativeP'] = False

    def __str__(self):
        return 'Newton method with MUMPS LU solver'

    @staticmethod
    def setup_parser_options(parser):
        gs.GeneralSolver.setup_parser_options(parser)

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
            problem.save_vel(False, u0)

        # boundary conditions
        bcu, bcp = problem.get_boundary_conditions(False, self.W.sub(0), self.W.sub(1))

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
        # prm['newton_solver']['lu_solver']['same_nonzero_pattern'] = True

        info(NS_solver.parameters, True)

        self.tc.end('init')

        # Time-stepping
        info("Running of direct method")
        ttime = self.metadata['time']
        t = dt
        step = 1
        while t < (ttime + dt/2.0):
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
                problem.save_vel(False, velSp)
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
            problem.compute_functionals(u, p, t, step)

            # Move to next time step
            self.tc.start('next')
            u0.assign(velSp)
            t = round(t + dt, 6)  # round time step to 0.000001
            step += 1
            self.tc.end('next')

        info("Finished: direct method")
        problem.report()
        return 0
