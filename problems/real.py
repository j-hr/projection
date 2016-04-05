from __future__ import print_function
from dolfin import assemble, interpolate, Expression, Function, DirichletBC, norm, errornorm, Constant
from dolfin.cpp.common import toc, mpi_comm_world, DOLFIN_EPS
from dolfin.cpp.function import near
from dolfin.cpp.io import HDF5File
from dolfin.cpp.mesh import Mesh, MeshFunction
from ufl import Measure, dx, cos, sin, FacetNormal, inner, grad, outer, Identity, sym
from math import pi, sqrt

from problems import general_problem as gp
import womersleyBC


class Problem(gp.GeneralProblem):
    def __init__(self, args, tc, metadata):
        self.has_analytic_solution = False
        self.problem_code = 'REAL'
        super(Problem, self).__init__(args, tc, metadata)

        self.name = 'test on real mesh'
        self.status_functional_str = 'not selected'

        # input parameters
        self.ic = args.ic
        self.factor = args.factor
        self.scale_factor.append(self.factor)

        self.nu = 3.71 * args.nu  # kinematic viscosity

        # Import gmsh mesh
        self.compatible_meshes = ['HYK']
        if args.mesh not in self.compatible_meshes:
            exit('Bad mesh, should be some from %s' % str(self.compatible_meshes))

        self.mesh = Mesh("meshes/" + args.mesh + ".xml")
        self.cell_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_physical_region.xml")
        self.facet_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_facet_region.xml")
        # self.dsIn = Measure("ds", subdomain_id=2, subdomain_data=self.facet_function)
        # self.dsOut = Measure("ds", subdomain_id=3, subdomain_data=self.facet_function)
        # self.dsWall = Measure("ds", subdomain_id=1, subdomain_data=self.facet_function)
        self.normal = FacetNormal(self.mesh)
        print("Mesh name: ", args.mesh, "    ", self.mesh)
        print("Mesh norm max: ", self.mesh.hmax())
        print("Mesh norm min: ", self.mesh.hmin())

        self.actual_time = None
        self.v_in_2 = None
        self.v_in_2_normal = [0.0, 1.0, 0.0]
        self.v_in_2_center = [1.59128, -13.6391, 7.24912]
        self.v_in_2_r = 1.01077
        self.v_in_4 = None
        self.v_in_4_normal = [0.1, -1.0, -0.37]
        self.v_in_4_center = [-4.02584, 7.70146, 8.77694]
        self.v_in_4_r = 0.553786

    def __str__(self):
        return 'test on real mesh'

    @staticmethod
    def setup_parser_options(parser):
        super(Problem, Problem).setup_parser_options(parser)
        parser.add_argument('--ic', help='Initial condition', choices=['zero'], default='zero')
        parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)

    def initialize(self, V, Q, PS, D):
        super(Problem, self).initialize(V, Q, PS, D)

        print("IC type: " + self.ic)
        print("Velocity scale factor = %4.2f" % self.factor)

        self.v_in_2 = Problem.InputVelocityProfile(self.factor, self.v_in_2_center, self.v_in_2_normal, self.v_in_2_r)
        self.v_in_4 = Problem.InputVelocityProfile(self.factor, self.v_in_4_center, self.v_in_4_normal, self.v_in_4_r)

    class InputVelocityProfile(Expression):
        def __init__(self, factor, center, normal, radius, **kwargs):
            # super(Expression, self).__init__()
            super(Problem.InputVelocityProfile, self).__init__(**kwargs)
            self.t = 0.0
            self.factor = factor
            self.center = center
            self.radius = radius
            self.normal = normal

        def eval(self, value, x):
            x_dist2 = float((x[0]-self.center[0])*(x[0]-self.center[0]))
            y_dist2 = float((x[1]-self.center[1])*(x[1]-self.center[1]))
            z_dist2 = float((x[2]-self.center[2])*(x[2]-self.center[2]))
            rad = float(sqrt(x_dist2+y_dist2+z_dist2))
            # do not evaluate on boundaries or outside of circle:
            velocity = 0 if near(rad, self.radius) or rad > self.radius else \
                2*Problem.v_function(self.t)*(self.radius - rad)/self.radius   # QQ je centerline 2*prumerna?
            value[0] = velocity * self.normal[0]
            value[1] = velocity * self.normal[1]
            value[2] = velocity * self.normal[2]

        def value_shape(self):
            return (3,)

    # jde o prumernou rychlost, nikoliv centerline
    @staticmethod
    def v_function(tt):
        # nejprve zadavane casti
        T = 1.0  # velikost jedne periody v s
        v_m = 300  # min.rychlost
        v_M = 800  # max.rychlost

        # pocitane koeficienty pro funkce
        a_1 = (-36)*(v_M-v_m)/T/T
        a_2 = (-12)*(v_M-v_m)/T/T
        a_3 = 3*(v_M-v_m)/T/T
        # a_4=-a_3;

        # vysledky funkce
        t = tt % T
        if t < T / 6.0:
            return v_M + (a_1*(t-(T/6.0))*(t-(T/6.0)))
        elif t < T / 3.0:
            return v_M + (a_2*(t-(T/6.0))*(t-(T/6.0)))
        elif t < 2.0 * T / 3.0:
            return v_m + (v_M-v_m)/3.0 + (a_3*(t-(2.0*T/3.0))*(t-(2.0*T/3.0)))
        else:
            return v_m + (v_M-v_m)/3.0 - (a_3*(t-(2.0*T/3.0))*(t-(2.0*T/3.0)))

    def get_boundary_conditions(self, use_pressure_BC):
        # boundary parts: 1 walls, 2, 4 inflow, 3, 5 outflow
        # Boundary conditions
        bc0 = DirichletBC(self.vSpace, (0.0, 0.0, 0.0), self.facet_function, 1)
        inflow2 = DirichletBC(self.vSpace, self.v_in_2, self.facet_function, 2)
        inflow4 = DirichletBC(self.vSpace, self.v_in_4, self.facet_function, 4)
        bcu = [inflow2, inflow4, bc0]
        bcp = []
        if use_pressure_BC:
            outflow = DirichletBC(self.pSpace, 0.0, self.facet_function, 5)  # QQ or 3 or both?
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

    def update_time(self, actual_time):
        super(Problem, self).update_time(actual_time)
        if self.actual_time > 0.5 and int(round(self.actual_time * 1000)) % 1000 == 0:
            self.isWholeSecond = True
            seconds = int(round(self.actual_time))
            self.second_list.append(seconds)
            self.N1 = seconds*self.stepsInSecond
            self.N0 = (seconds-1)*self.stepsInSecond
        else:
            self.isWholeSecond = False

        # Update boundary condition
        self.tc.start('updateBC')
        self.v_in_2.t = actual_time
        self.v_in_4.t = actual_time
        self.tc.end('updateBC')

    def save_pressure(self, is_tent, pressure):
        super(Problem, self).save_pressure(is_tent, pressure)
