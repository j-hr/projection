from __future__ import print_function

from dolfin import assemble, Expression, Function, DirichletBC, plot, interpolate
from dolfin.cpp.common import info
from dolfin.cpp.function import near
from dolfin.cpp.mesh import Mesh, MeshFunction, FacetFunction, vertices, facets
from math import sqrt
from ufl import Measure, FacetNormal, inner, ds, div, transpose, grad, dx, sym

from problems import general_problem as gp


class Problem(gp.GeneralProblem):
    def __init__(self, args, tc, metadata):
        self.has_analytic_solution = False
        self.problem_code = 'REAL'
        super(Problem, self).__init__(args, tc, metadata)

        self.name = 'test on real mesh'
        self.status_functional_str = 'outflow/inflow'
        self.last_inflow = 0

        # input parameters
        self.ic = args.ic
        self.factor = args.factor
        self.metadata['factor'] = self.factor
        self.scale_factor.append(self.factor)

        self.nu = 3.71 * args.nu  # kinematic viscosity

        # Import mesh
        self.compatible_meshes = ['HYK']
        if args.mesh not in self.compatible_meshes:
            exit('Bad mesh, should be some from %s' % str(self.compatible_meshes))
        self.mesh, self.facet_function = super(Problem, self).loadMesh(args.mesh)
        info("Mesh name: " + args.mesh + "    " + str(self.mesh))
        # self.dsWall = Measure("ds", subdomain_id=1, subdomain_data=self.facet_function)
        self.mesh_volume = 564.938845339
        self.normal = FacetNormal(self.mesh)
        self.dsOut1 = Measure("ds", subdomain_id=3, subdomain_data=self.facet_function)
        self.dsOut2 = Measure("ds", subdomain_id=5, subdomain_data=self.facet_function)
        self.dsIn1 = Measure("ds", subdomain_id=2, subdomain_data=self.facet_function)
        self.dsIn2 = Measure("ds", subdomain_id=4, subdomain_data=self.facet_function)
        self.listDict.update({
            'outflow': {'list': [], 'name': 'outflow rate', 'abrev': 'OUT', 'slist': []},
            'outflow1': {'list': [], 'name': 'outflow rate back', 'abrev': 'OUT1', 'slist': []},
            'outflow2': {'list': [], 'name': 'outflow rate front', 'abrev': 'OUT2', 'slist': []},
            'inflow': {'list': [], 'name': 'inflow rate', 'abrev': 'IN', 'slist': []},
            'oiratio': {'list': [], 'name': 'outflow/inflow ratio (mass conservation)', 'abrev': 'O/I', 'slist': []},
        })

        self.can_force_outflow = True

        self.actual_time = None
        self.v_in_2 = None
        self.v_in_2_normal = [0.0, 1.0, 0.0]
        self.v_in_2_center = [1.59128, -13.6391, 7.24912]
        self.v_in_2_r = 1.01077
        self.v_in_4 = None
        self.v_in_4_normal = [0.1, -1.0, -0.37]
        self.v_in_4_center = [-4.02584, 7.70146, 8.77694]
        self.v_in_4_r = 0.553786

    # TODO move to general using get_outflow_measures()
    def compute_outflow(self, velocity):
        out = assemble(inner(velocity, self.normal)*self.dsOut1 + inner(velocity, self.normal)*self.dsOut2)
        return out

    def get_outflow_measures(self):
        return [self.dsOut1, self.dsOut2]

    def get_outflow_measure_form(self):
        return self.dsOut1 + self.dsOut2

    def __str__(self):
        return 'test on real mesh'

    @staticmethod
    def setup_parser_options(parser):
        super(Problem, Problem).setup_parser_options(parser)
        parser.add_argument('--ic', help='Initial condition', choices=['zero'], default='zero')
        parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)

    def initialize(self, V, Q, PS, D):
        super(Problem, self).initialize(V, Q, PS, D)

        info("IC type: " + self.ic)
        info("Velocity scale factor = %4.2f" % self.factor)

        self.v_in_2 = Problem.InputVelocityProfile(self.factor, self.v_in_2_center, self.v_in_2_normal, self.v_in_2_r)
        self.v_in_4 = Problem.InputVelocityProfile(self.factor, self.v_in_4_center, self.v_in_4_normal, self.v_in_4_r)

        # TODO move to general using get_outflow_measures method
        one = (interpolate(Expression('1.0'), Q))
        self.outflow_area = assemble(one*self.dsOut1 + one*self.dsOut2)
        info('Outflow area: %f' % self.outflow_area)

    class InputVelocityProfile(Expression):
        def __init__(self, factor, center, normal, radius, **kwargs):
            # super(Expression, self).__init__()
            super(Problem.InputVelocityProfile, self).__init__(**kwargs)
            self.t = 0.
            self.onset_factor = 1.
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
                2.*self.onset_factor*self.factor*Problem.v_function(self.t)*(1.0 - rad*rad/(self.radius*self.radius))   # QQ je centerline 2*prumerna?
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
            outflow3 = DirichletBC(self.pSpace, 0.0, self.facet_function, 3)  # QQ or 3 or both?
            outflow5 = DirichletBC(self.pSpace, 0.0, self.facet_function, 5)  # QQ or 3 or both?
            bcp = [outflow3, outflow5]
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
        self.tc.start('updateBC')
        self.v_in_2.t = actual_time
        self.v_in_2.onset_factor = self.onset_factor
        self.v_in_4.t = actual_time
        self.v_in_4.onset_factor = self.onset_factor
        in1 = assemble(inner(self.v_in_2, self.normal)*self.dsIn1)
        in2 = assemble(inner(self.v_in_4, self.normal)*self.dsIn2)
        self.last_inflow = in1+in2
        info('Inflow: %f' % self.last_inflow)
        self.listDict['inflow']['list'].append(self.last_inflow)

        self.tc.end('updateBC')

    def save_pressure(self, is_tent, pressure):
        super(Problem, self).save_pressure(is_tent, pressure)

    def save_vel(self, is_tent, field, t):
        super(Problem, self).save_vel(is_tent, field, t)

    def compute_functionals(self, velocity, pressure, t):
        super(Problem, self).compute_functionals(velocity, pressure, t)
        out1 = assemble(inner(velocity, self.normal)*self.dsOut1)
        out2 = assemble(inner(velocity, self.normal)*self.dsOut2)
        out = out1 + out2
        self.listDict['outflow1']['list'].append(out1)
        self.listDict['outflow2']['list'].append(out2)
        self.listDict['outflow']['list'].append(out)
        info('Outflow: %f' % out)
        self.last_status_functional = out/abs(self.last_inflow)
        self.listDict['oiratio']['list'].append(self.last_status_functional)
        info('Outflow/Inflow: %f' % self.last_status_functional)

