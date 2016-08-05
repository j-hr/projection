from __future__ import print_function
from dolfin import assemble, interpolate, Expression, Function, DirichletBC, norm, errornorm, Constant, FunctionSpace, \
    plot
from dolfin.cpp.common import toc, mpi_comm_world, DOLFIN_EPS
from dolfin.cpp.function import near
from dolfin.cpp.io import HDF5File
from dolfin.cpp.mesh import Mesh, MeshFunction
from ufl import Measure, dx, cos, sin, FacetNormal, inner, grad, outer, Identity, sym
from math import pi, sqrt

from problems import general_problem as gp
import womersleyBC


# Note: main unit is 1 m
class Problem(gp.GeneralProblem):
    def __init__(self, args, tc, metadata):
        self.has_analytic_solution = False
        self.problem_code = 'FACB'
        super(Problem, self).__init__(args, tc, metadata)

        self.name = 'test on real mesh'
        self.status_functional_str = 'not selected'

        # input parameters
        self.factor = args.factor
        self.scale_factor.append(self.factor)

        self.nu = 0.001 * args.nufactor  # kinematic viscosity

        # Import gmsh mesh
        self.compatible_meshes = ['bench3D_1', 'bench3D_2', 'bench3D_3']
        if args.mesh not in self.compatible_meshes:
            exit('Bad mesh, should be some from %s' % str(self.compatible_meshes))

        self.mesh = Mesh("meshes/" + args.mesh + ".xml")
        self.cell_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_physical_region.xml")
        self.facet_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_facet_region.xml")
        self.dsIn = Measure("ds", subdomain_id=2, subdomain_data=self.facet_function)
        self.dsOut = Measure("ds", subdomain_id=3, subdomain_data=self.facet_function)
        self.dsWall = Measure("ds", subdomain_id=1, subdomain_data=self.facet_function)
        self.dsCyl = Measure("ds", subdomain_id=5, subdomain_data=self.facet_function)
        self.normal = FacetNormal(self.mesh)
        print("Mesh name: ", args.mesh, "    ", self.mesh)
        print("Mesh norm max: ", self.mesh.hmax())
        print("Mesh norm min: ", self.mesh.hmin())

        self.actual_time = None
        self.v_in = None

    def __str__(self):
        return 'flow around cylinder 3D benchmark'

    @staticmethod
    def setup_parser_options(parser):
        super(Problem, Problem).setup_parser_options(parser)
        parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)
        parser.add_argument('--nufactor', help='kinematic viscosity factor', type=float, default=1.0)

    def initialize(self, V, Q, PS, D):
        super(Problem, self).initialize(V, Q, PS, D)

        print("Velocity scale factor = %4.2f" % self.factor)

        self.v_in = Problem.InputVelocityProfile(self.factor)
        # plot(interpolate(self.v_in, self.vSpace), interactive=True)
        # scalarSpace = FunctionSpace(self.mesh, "Lagrange", 2)
        # area = assemble(interpolate(Expression("1.0"), scalarSpace) * self.dsIn)
        # expr = Expression("factor*0.45*(x[1]*(0.41-x[1]) * x[2]*(0.41-x[2]))/(0.205*0.205*0.205*0.205)",
        #                   factor=self.factor)
        # average = assemble((1.0/area) * interpolate(expr, scalarSpace) * self.dsIn)
        # print('Average velocity:', average)
        # re = average * 0.1 / self.nu  # average_velocity*cylinger_diameter/kinematic_viscosity
        re = 20.0 * self.factor / self.args.nufactor  # average_velocity*cylinger_diameter/kinematic_viscosity
        print('Reynolds number:', re)

        one = (interpolate(Expression('1.0'), Q))
        self.mesh_volume = assemble(one*dx)
        self.outflow_area = assemble(one*self.dsOut)
        print('Outflow area:', self.outflow_area)

    # parabolic profile on square normed so centerline velocity = 1m/s * factor
    class InputVelocityProfile(Expression):
        def __init__(self, factor):
            # super(Expression, self).__init__()
            super(Problem.InputVelocityProfile, self).__init__()
            self.factor = factor
            # self.t = 0.0

        def eval(self, value, x):
            value[0] = self.factor*0.45*(x[1]*(0.41-x[1]) * x[2]*(0.41-x[2]))/(0.205*0.205*0.205*0.205)
            value[1] = 0.0
            value[2] = 0.0

        def value_shape(self):
            return (3,)

    def get_boundary_conditions(self, use_pressure_BC, v_space, p_space):
        # boundary parts: 1 walls, 2, 4 inflow, 3, 5 outflow
        # Boundary conditions
        bc_wall = DirichletBC(v_space, (0.0, 0.0, 0.0), self.facet_function, 1)
        bc_cyl = DirichletBC(v_space, (0.0, 0.0, 0.0), self.facet_function, 5)
        inflow = DirichletBC(v_space, self.v_in, self.facet_function, 2)
        bcu = [inflow, bc_cyl, bc_wall]
        bcp = []
        if use_pressure_BC:
            outflow = DirichletBC(p_space, 0.0, self.facet_function, 3)
            bcp = [outflow]
        return bcu, bcp

    def get_initial_conditions(self, function_list):
        out = []
        for d in function_list:
            if d['type'] == 'v':
                f = Function(self.vSpace)
            if d['type'] == 'p':
                f = Function(self.pSpace)
            out.append(f)
        return out

    def update_time(self, actual_time, step_number):
        super(Problem, self).update_time(actual_time, step_number)
        if self.actual_time > 0.5 and int(round(self.actual_time * 1000)) % 1000 == 0:
            self.isWholeSecond = True
            seconds = int(round(self.actual_time))
            self.second_list.append(seconds)
            self.N1 = seconds*self.stepsInSecond
            self.N0 = (seconds-1)*self.stepsInSecond
        else:
            self.isWholeSecond = False

        # Update boundary condition
        # self.tc.start('updateBC')
        # self.v_in.t = actual_time
        # self.tc.end('updateBC')

    def save_pressure(self, is_tent, pressure):
        super(Problem, self).save_pressure(is_tent, pressure)
