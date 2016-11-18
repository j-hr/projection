from __future__ import print_function

import csv
import math
from dolfin import assemble, Expression, Function, DirichletBC
from dolfin.cpp.common import info
from dolfin.cpp.function import near
from math import sqrt
from ufl import Measure, FacetNormal, inner
from problems import general_problem as gp


class Problem(gp.GeneralProblem):
    def __init__(self, args, tc, metadata):
        self.has_analytic_solution = False
        self.problem_code = 'REAL'
        super(Problem, self).__init__(args, tc, metadata)

        self.name = 'test on real mesh'
        self.status_functional_str = 'outflow/inflow'
        self.last_inflow = 0

        # time settings
        self.itp_lengths = {1: 1.0, 2: 0.9375, }
        self.cycle_length = self.itp_lengths[self.args.itp]

        # input parameters
        self.nu = self.args.nu  # kinematic viscosity
        self.factor = args.factor
        self.metadata['factor'] = self.factor
        self.scale_factor.append(self.factor)

        self.tc.start('mesh')
        # Import mesh
        try:
            self.mesh, self.facet_function = super(Problem, self).loadMesh(args.mesh)
            info("Mesh name: " + args.mesh + "    " + str(self.mesh))
            f_ini = open('meshes/'+args.mesh+'.ini', 'r')
            reader = csv.reader(f_ini, delimiter=' ', escapechar='\\')
        except (EnvironmentError, RuntimeError):
            print('Unable to open mesh.hdf5 or mesh.ini file. Check if the mesh was prepared to be used '
                  'with \"real\" problem.')
            exit(1)

        # load inflows and outflows (interfaces) from mesh.ini file
        obj = None
        self.interfaces = []
        for row in reader:
            if not row:
                pass
            elif row[0] == 'volume':
                self.mesh_volume = float(row[1])
            elif row[0] == 'in':
                if obj is not None:
                    self.interfaces.append(obj)
                obj = {'inflow': True, 'number': row[1]}
            elif row[0] == 'out':
                if obj is not None:
                    self.interfaces.append(obj)
                obj = {'inflow': False, 'number': row[1]}
            else:
                if len(row) == 2:  # scalar values
                    obj[row[0]] = row[1]
                else:                # vector values
                    obj[row[0]] = [float(f) for f in row[1:]]
        self.interfaces.append(obj)
        f_ini.close()
        self.tc.end('mesh')

        # collect inflows and outflows into separate lists
        self.outflow_area = 0
        self.inflows = []
        self.outflows = []
        for obj in self.interfaces:
            if not obj['inflow']:
                self.outflow_area += float(obj['S'])
                self.outflows.append(obj)
            else:
                self.inflows.append(obj)
        info('Outflow area: %f' % self.outflow_area)

        # self.dsWall = Measure("ds", subdomain_id=1, subdomain_data=self.facet_function)
        self.normal = FacetNormal(self.mesh)

        # generate measures, collect measure lists
        self.inflow_measures = []
        for obj in self.interfaces:
            obj['measure'] = Measure("ds", subdomain_id=int(obj['number']), subdomain_data=self.facet_function)
            if obj['inflow']:
                self.inflow_measures.append(obj['measure'])
            else:
                self.outflow_measures.append(obj['measure'])

        self.listDict.update({
            'outflow': {'list': [], 'name': 'outflow rate', 'abrev': 'OUT', 'slist': []},
            'inflow': {'list': [], 'name': 'inflow rate', 'abrev': 'IN', 'slist': []},
            'oiratio': {'list': [], 'name': 'outflow/inflow ratio (mass conservation)', 'abrev': 'O/I', 'slist': []},
        })
        for obj in self.outflows:
            n = obj['number']
            self.listDict.update({'outflow' + n:
                                  {'list': [], 'name': 'outflow rate ' + n, 'abrev': 'OUT' + n, 'slist': []}})
        self.can_force_outflow = True

    def get_outflow_measures(self):
        return self.outflow_measures

    def __str__(self):
        return 'test on real mesh'

    @staticmethod
    def setup_parser_options(parser):
        super(Problem, Problem).setup_parser_options(parser)
        #parser.add_argument('--ic', help='Initial condition', choices=['zero'], default='zero')
        parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)
        parser.add_argument('--nu', help='kinematic viscosity factor', type=float, default=3.71)
        parser.add_argument('--itp', help='inflow time profile polynomial', choices=[1, 2], default=2)

    def initialize(self, V, Q, PS, D):
        super(Problem, self).initialize(V, Q, PS, D)
        #info("IC type: " + self.ic)
        info("Velocity scale factor = %4.2f" % self.factor)
        itp = Problem.v_function if self.args.itp == 1 else Problem.v_function_2
        # generate inflow profiles
        for obj in self.inflows:
            obj['velocity_profile'] = Problem.InputVelocityProfile(self.factor*float(obj['reference_coef']),
                                                                   obj['center'], obj['normal'], float(obj['radius']),
                                                                   itp, degree=2)

    class InputVelocityProfile(Expression):
        def __init__(self, factor, center, normal, radius, itp, **kwargs):
            self.t = 0.
            self.onset_factor = 1.
            self.factor = factor
            self.center = center
            self.radius = radius
            self.normal = normal
            self.itp = itp

        def eval(self, value, x):
            x_dist2 = float((x[0]-self.center[0])*(x[0]-self.center[0]))
            y_dist2 = float((x[1]-self.center[1])*(x[1]-self.center[1]))
            z_dist2 = float((x[2]-self.center[2])*(x[2]-self.center[2]))
            rad = float(sqrt(x_dist2+y_dist2+z_dist2))
            # do not evaluate on boundaries or outside of circle:
            velocity = 0 if near(rad, self.radius) or rad > self.radius else \
                2.*self.onset_factor*self.factor*self.itp(self.t)*(1.0 - rad*rad/(self.radius*self.radius))
            value[0] = velocity * self.normal[0]
            value[1] = velocity * self.normal[1]
            value[2] = velocity * self.normal[2]

        def value_shape(self):
            return (3,)

    # polynomial defining average inflow velocity
    @staticmethod
    def v_function(tt):
        T = 1.0  # Period length
        v_m = 300  # min. velocity
        v_M = 800  # max. velocity

        a_1 = (-36)*(v_M-v_m)/T/T
        a_2 = (-12)*(v_M-v_m)/T/T
        a_3 = 3*(v_M-v_m)/T/T

        t = tt % T
        if t < T / 6.0:
            return v_M + (a_1*(t-(T/6.0))*(t-(T/6.0)))
        elif t < T / 3.0:
            return v_M + (a_2*(t-(T/6.0))*(t-(T/6.0)))
        elif t < 2.0 * T / 3.0:
            return v_m + (v_M-v_m)/3.0 + (a_3*(t-(2.0*T/3.0))*(t-(2.0*T/3.0)))
        else:
            return v_m + (v_M-v_m)/3.0 - (a_3*(t-(2.0*T/3.0))*(t-(2.0*T/3.0)))

    # another polynomial defining average inflow velocity
    @staticmethod
    def v_function_2(tt):
        T = 0.9375  # Period length
        v_m = 180  # min. velocity
        v_M = 480  # max. velocity

        a_12 = (-36.0) * (v_M - v_m) / T / T
        a_22 = (-24.0) * (v_M - v_m) / T / T
        a_32 = 0.5 * (v_M - v_m) / T / T

        # tt -= T/2.
        t = tt % T
        # t = tt
        if t < T / 6.0:
            return v_M + (a_12*(t-(T/6))*(t-(T/6)))
        elif t < T / 3.0:
            return v_M + (a_22*(t-(T/6))*(t-(T/6)))
        else:
            return v_m + (a_32*(t-T)*(t-T))

    def get_boundary_conditions(self, use_pressure_BC, v_space, p_space):
        # boundary parts: 1 walls, inflows and outflows specified in [meshName].ini file
        bc0 = DirichletBC(v_space, (0.0, 0.0, 0.0), self.facet_function, 1)   # no-slip
        bcu = [bc0]
        for obj in self.inflows:
            bcu.append(DirichletBC(v_space, obj['velocity_profile'], self.facet_function, int(obj['number'])))
        bcp = []
        if use_pressure_BC:
            for obj in self.outflows:
                bcp.append(DirichletBC(p_space, 0., self.facet_function, int(obj['number'])))
        return bcu, bcp

    def get_initial_conditions(self, function_list):
        out = []
        for d in function_list:
            if d['type'] == 'v':
                f = Function(self.vSpace)  # zero velocity
            if d['type'] == 'p':
                f = Function(self.pSpace)  # zero pressure
            out.append(f)
        return out

    def update_time(self, actual_time, step_number):
        super(Problem, self).update_time(actual_time, step_number)
        if self.actual_time > 0.5 and abs(math.modf(actual_time)[0]) < 0.5*self.metadata['dt']:
            self.second_list.append(int(round(self.actual_time)))

        # Update boundary condition
        self.tc.start('updateBC')
        self.last_inflow = 0
        for obj in self.inflows:
            obj['velocity_profile'].t = actual_time
            obj['velocity_profile'].onset_factor = self.onset_factor
            self.last_inflow += assemble(inner(obj['velocity_profile'], self.normal)*obj['measure'])
        info('Inflow: %f' % self.last_inflow)
        self.listDict['inflow']['list'].append(self.last_inflow)

        self.tc.end('updateBC')

    def save_pressure(self, is_tent, pressure):
        super(Problem, self).save_pressure(is_tent, pressure)

    def save_vel(self, is_tent, field):
        super(Problem, self).save_vel(is_tent, field)

    def compute_functionals(self, velocity, pressure, t, step):
        super(Problem, self).compute_functionals(velocity, pressure, t, step)
        out = 0
        for obj in self.outflows:
            outflow = assemble(inner(velocity, self.normal)*obj['measure'])
            out += outflow
            self.listDict['outflow'+obj['number']]['list'].append(outflow)
        self.listDict['outflow']['list'].append(out)
        info('Outflow: %f' % out)
        self.last_status_functional = out/abs(self.last_inflow)
        self.listDict['oiratio']['list'].append(self.last_status_functional)
        info('Outflow/Inflow: %f' % self.last_status_functional)

