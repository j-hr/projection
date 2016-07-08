from __future__ import print_function
import os, sys, traceback
import csv, cPickle
from dolfin import Function, TestFunction, assemble, interpolate, Expression, project, norm, errornorm, TensorFunctionSpace, plot, \
    FunctionSpace, VectorFunctionSpace
from dolfin.cpp.common import mpi_comm_world, toc, MPI, info, begin, end
from dolfin.cpp.io import XDMFFile, HDF5File
from dolfin.cpp.mesh import Mesh, MeshFunction, SubMesh, BoundaryMesh, Cell
from ufl import dx, div, inner, grad, sym, transpose, sqrt as sqrt_ufl, Identity, FacetNormal, dot, \
    FacetArea, ds, as_vector
from math import sqrt, pi, cos, modf
import numpy as np

class GeneralProblem(object):
    def __init__(self, args, tc, metadata):
        self.MPI_rank = MPI.rank(mpi_comm_world())

        self.metadata = metadata

        # need to be specified in subclass init before calling this init
        self.problem_code = self.problem_code
        self.metadata['pcode'] = self.problem_code
        self.has_analytic_solution = self.has_analytic_solution
        self.metadata['hasAnalyticSolution'] = self.has_analytic_solution

        self.args = args
        self.tc = tc
        self.tc.init_watch('mesh', 'mesh import', True)
        self.tc.init_watch('saveP', 'Saved pressure', True)
        self.tc.init_watch('saveVel', 'Saved velocity', True)
        self.tc.init_watch('averageP', 'Averaged pressure', True)
        self.tc.init_watch('updateBC', 'Updated velocity BC', True)
        self.tc.init_watch('div', 'Computed and saved divergence', True)
        self.tc.init_watch('divNorm', 'Computed norm of divergence', True)
        self.tc.init_watch('WSSinit', 'Initialized mesh for WSS', False)
        self.tc.init_watch('WSS', 'Computed and saved WSS', True)

        # If it is sensible (and implemented) to force pressure gradient on outflow boundary
        # 1. set self.outflow_area in initialize
        # 2. implement self.compute_outflow and get_outflow_measures
        self.can_force_outflow = False
        self.outflow_area = None
        self.normal = None
        self.mesh = None
        self.facet_function = None
        self.mesh_volume = None
        self.outflow_measures = []

        # stopping criteria (for relative H1 velocity error norm) (if known)
        self.divergence_treshold = 10

        # used for writing status files .run to monitor progress during computation:
        self.last_status_functional = 0.0
        self.status_functional_str = 'to be defined in Problem class'

        self.stepsInSecond = None
        self.volume = None
        self.vSpace = None
        self.vFunction = None
        self.divSpace = None
        self.divFunction = None
        self.pSpace = None
        self.pFunction = None
        self.solutionSpace = None
        self.solution = None

        self.actual_time = 0.0
        self.step_number = 0
        self.save_this_step = False
        self.isWholeSecond = None
        self.N1 = None
        self.N0 = None

        self.vel_normalization_factor = []
        self.pg_normalization_factor = []
        self.p_normalization_factor = []
        self.scale_factor = []

        self.analytic_v_norm_L2 = None
        self.analytic_v_norm_H1 = None
        self.analytic_v_norm_H1w = None

        # for WSS generation
        metadata['hasWSS'] = (args.wss != 'none')
        metadata['WSSmethod'] = self.args.wss_method
        self.nu = 0.0
        self.T = None
        self.Tb = None
        self.wall_mesh = None
        self.wall_mesh_oriented = None
        self.Vb = None
        self.Sb = None
        self.VDG = None
        self.R = None
        self.SDG = None
        self.nb = None
        self.peak_time_steps = None

        # dictionaries for output files
        self.fileDict = {'u': {'name': 'velocity'},
                         'p': {'name': 'pressure'},
                         # 'pg': {'name': 'pressure_grad'},
                         'd': {'name': 'divergence'}}
        self.fileDictTent = {'u2': {'name': 'velocity_tent'},
                             'd2': {'name': 'divergence_tent'}}
        self.fileDictDiff = {'uD': {'name': 'velocity_diff'},
                             'pD': {'name': 'pressure_diff'},
                             'pgD': {'name': 'pressure_grad_diff'}}
        self.fileDictTentDiff = {'u2D': {'name': 'velocity_tent_diff'}}
        self.fileDictTentP = {'p2': {'name': 'pressure_tent'}}
        #                      'pg2': {'name': 'pressure_grad_tent'}}
        self.fileDictTentPDiff = {'p2D': {'name': 'pressure_tent_diff'}}
        #                          'pg2D': {'name': 'pressure_grad_tent_diff'}}
        self.fileDictLDSG = {'ldsg': {'name': 'ldsg'},
                             'ldsg2': {'name': 'ldsg_tent'}}
        self.fileDictWSS = {'wss': {'name': 'wss'}, }
        self.fileDictWSSnorm = {'wss_norm': {'name': 'wss_norm'}, }

        # lists of functionals and other scalar output data
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.second_list = []
        self.listDict = {}  # list of fuctionals
        # dictionary of data lists {list, name, abbreviation, add scaled row to report}
        # normalisation coefficients (time-independent) are added to some lists to be used in normalized data series
        #   coefficients are lists (updated during initialisation, so we cannot use float type)
        #   coefs are equal to average of respective value of analytic solution
        # norm lists (time-dependent normalisation coefficients) are added to some lists to be used in relative data
        #  series (to remove natural pulsation of error due to change in volume flow rate)
        # slist - lists for cycle-averaged values
        # L2(0) means L2 difference of pressures taken with zero average
        self.listDict = {
            'd': {'list': [], 'name': 'corrected velocity L2 divergence', 'abrev': 'DC', 'scale': self.scale_factor, 'slist': []},
            'd2': {'list': [], 'name': 'tentative velocity L2 divergence', 'abrev': 'DT', 'scale': self.scale_factor, 'slist': []},
            'pg': {'list': [], 'name': 'computed pressure gradient', 'abrev': 'PG', 'scale': self.scale_factor,
                   'norm': self.pg_normalization_factor},
            'pg2': {'list': [], 'name': 'computed pressure tent gradient', 'abrev': 'PTG', 'scale': self.scale_factor,
                    'norm': self.pg_normalization_factor},
            # 'dsg-l': {'list': [], 'name': 'div(2sym(grad(u))-laplace(u))', 'abrev': 'DSG-L'},  # div(2sym(grad(u))-laplace(u))
            # 'dgt': {'list': [], 'name': 'div(transpose(grad(u)))', 'abrev': 'DGT'},  # div(transpose(grad(u)))
        }
        if self.has_analytic_solution:
            self.listDict.update({
                'u_L2': {'list': [], 'name': 'corrected velocity L2 error', 'abrev': 'CE_L2', 'scale': self.scale_factor,
                         'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
                'u2L2': {'list': [], 'name': 'tentative velocity L2 error', 'abrev': 'TE_L2', 'scale': self.scale_factor,
                         'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
                'u_L2test': {'list': [], 'name': 'test corrected L2 velocity error', 'abrev': 'TestCE_L2', 'scale': self.scale_factor},
                'u2L2test': {'list': [], 'name': 'test tentative L2 velocity error', 'abrev': 'TestTE_L2', 'scale': self.scale_factor},
                'u_H1': {'list': [], 'name': 'corrected velocity H1 error', 'abrev': 'CE_H1', 'scale': self.scale_factor,
                         'relative': 'av_norm_H1', 'slist': []},
                'u2H1': {'list': [], 'name': 'tentative velocity H1 error', 'abrev': 'TE_H1', 'scale': self.scale_factor,
                         'relative': 'av_norm_H1', 'slist': []},
                'u_H1test': {'list': [], 'name': 'test corrected H1 velocity error', 'abrev': 'TestCE_H1', 'scale': self.scale_factor},
                'u2H1test': {'list': [], 'name': 'test tentative H1 velocity error', 'abrev': 'TestTE_H1', 'scale': self.scale_factor},
                'apg': {'list': [], 'name': 'analytic pressure gradient', 'abrev': 'APG', 'scale': self.scale_factor,
                        'norm': self.pg_normalization_factor},
                'av_norm_L2': {'list': [], 'name': 'analytic velocity L2 norm', 'abrev': 'AVN_L2'},
                'av_norm_H1': {'list': [], 'name': 'analytic velocity H1 norm', 'abrev': 'AVN_H1'},
                'ap_norm': {'list': [], 'name': 'analytic pressure norm', 'abrev': 'APN'},
                'p': {'list': [], 'name': 'pressure L2(0) error', 'abrev': 'PE', 'scale': self.scale_factor, 'slist': [],
                      'norm': self.p_normalization_factor},
                'pgE': {'list': [], 'name': 'computed pressure gradient error', 'abrev': 'PGE', 'scale': self.scale_factor,
                        'norm': self.pg_normalization_factor, 'slist': []},
                'pgEA': {'list': [], 'name': 'computed absolute pressure gradient error', 'abrev': 'PGEA',
                         'scale': self.scale_factor, 'norm': self.pg_normalization_factor},
                'p2': {'list': [], 'name': 'pressure tent L2(0) error', 'abrev': 'PTE', 'scale': self.scale_factor,
                       'slist': [], 'norm': self.p_normalization_factor},
                'pgE2': {'list': [], 'name': 'computed tent pressure tent gradient error', 'abrev': 'PTGE',
                         'scale': self.scale_factor, 'norm': self.pg_normalization_factor, 'slist': []},
                'pgEA2': {'list': [], 'name': 'computed absolute pressure tent gradient error',
                          'abrev': 'PTGEA', 'scale': self.scale_factor, 'norm': self.pg_normalization_factor}
            })

        # parse arguments
        self.nu_factor = args.nu
        self.onset = args.onset
        self.onset_factor = 0

        self.doSave = False
        self.saveOnlyVel = False
        self.doSaveDiff = False
        self.save_nth = args.savespace
        option = args.save
        if option == 'doSave' or option == 'diff' or option == 'only_vel':
            self.doSave = True
            if option == 'diff':
                self.doSaveDiff = True
                info('Saving velocity differences.')
            if option == 'only_vel':
                self.saveOnlyVel = True
                info('Saving only velocity profiles.')
            info('Saving solution ON.')
        elif option == 'noSave':
            self.doSave = False
            info('Saving solution OFF.')

        self.doErrControl = None
        self.testErrControl = False
        if args.error == "noEC":
            self.doErrControl = False
            info("Error control omitted")
        else:
            self.doErrControl = True
            if args.error == "test":
                self.testErrControl = True
                info("Error control in testing mode")
            else:
                info("Error control on")

        self.str_dir_name = "%s_%s_results" % (self.problem_code, metadata['name'])
        self.metadata['dir'] = self.str_dir_name
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name) and self.MPI_rank == 0:
            os.mkdir(self.str_dir_name)

    @staticmethod
    def setup_parser_options(parser):
        parser.add_argument('-e', '--error', help='Error control mode', choices=['doEC', 'noEC', 'test'], default='doEC')
        parser.add_argument('-S', '--save', help='Save solution mode', choices=['doSave', 'noSave', 'diff', 'only_vel'],
                            default='noSave')
        #   doSave: create .xdmf files with velocity, pressure, divergence
        #   diff: save also difference vel-sol
        #   noSave: do not create .xdmf files with velocity, pressure, divergence
        parser.add_argument('--savespace', help='save only n-th step in first cycle', type=int, default=1)
        parser.add_argument('--ST', help='save only n-th step in first cycle', choices=['min', 'peak', 'no_restriction'], default='no_restriction')
        # stronger than --savespace, to be used instead of it
        parser.add_argument('--nu', help='kinematic viscosity factor', type=float, default=1.0)
        parser.add_argument('--onset', help='boundary condition onset length', type=float, default=0.0)
        parser.add_argument('--ldsg', help='save laplace(u) - div(2sym(grad(u))) difference', action='store_true')
        parser.add_argument('--wss_method', help='compute wall shear stress', choices=['expression', 'integral'], default='integral')
        # expression does not work for too many processors (24 procs for 'HYK' is OK, 48 is too much)
        parser.add_argument('--wss', help='compute wall shear stress', choices=['none', 'all', 'peak'], default='none')
        # NT to use wss -S must be at least only_vel (doSave must be True for wss work properly)

    @staticmethod
    def loadMesh(mesh):
        f = HDF5File(mpi_comm_world(), 'meshes/'+mesh+'.hdf5', 'r')
        mesh = Mesh()
        f.read(mesh, 'mesh', False)
        facet_function = MeshFunction("size_t", mesh)
        f.read(facet_function, 'facet_function')
        return mesh, facet_function

    def initialize(self, V, Q, PS, D):
        self.vSpace = V
        self.divSpace = D
        self.pSpace = Q
        self.solutionSpace = V
        self.vFunction = Function(V)
        self.divFunction = Function(D)
        self.pFunction = Function(Q)
        self.volume = assemble(interpolate(Expression("1.0"), Q) * dx)

        if self.doSave:
            # self.pgSpace = VectorFunctionSpace(mesh, "DG", 0)
            # self.pgFunction = Function(self.pgSpace)
            self.initialize_xdmf_files()
        self.stepsInSecond = int(round(1.0 / self.metadata['dt']))
        info('stepsInSecond = %d' % self.stepsInSecond)

        if self.args.ST == 'min' or self.args.wss == 'peak':
            # NT manualy written here:
            chosen_steps = [0.1, 0.125, 0.15, 0.175, 0.186, 0.187, 0.188, 0.189, 0.190, 0.2]
            self.peak_time_steps = [int(round(chosen / self.metadata['dt'])) for chosen in chosen_steps]
            for ch in self.peak_time_steps:
                info('Chosen peak time steps at every %dth step in %d steps' % (ch, self.stepsInSecond))

        if self.args.wss != 'none':
            # NT implemented in general_problem, but sensible only in womersley_cylinder and real
            # but everything about wss is ommited as long one does not use --wss '...'
            self.tc.start('WSSinit')
            if self.args.wss_method == 'expression':
                self.T = TensorFunctionSpace(self.mesh, 'Lagrange', 1)
                info('Generating boundary mesh')
                self.wall_mesh = BoundaryMesh(self.mesh, 'exterior')
                self.wall_mesh_oriented = BoundaryMesh(self.mesh, 'exterior', order=False)
                info('  Boundary mesh geometric dim: %d' % self.wall_mesh.geometry().dim())
                info('  Boundary mesh topologic dim: %d' % self.wall_mesh.topology().dim())
                self.Tb = TensorFunctionSpace(self.wall_mesh, 'Lagrange', 1)
                self.Vb = VectorFunctionSpace(self.wall_mesh, 'Lagrange', 1)
                info('Generating normal to boundary')
                # self.nb = self.get_facet_normal(self.wall_mesh)
                normal_expr = self.NormalExpression(self.wall_mesh_oriented)
                Vn = VectorFunctionSpace(self.wall_mesh, 'DG', 0)
                self.nb = project(normal_expr, Vn)
                self.Sb = FunctionSpace(self.wall_mesh, 'DG', 0)

            if self.args.wss_method == 'integral':
                self.SDG = FunctionSpace(self.mesh, 'DG', 0)

            self.tc.end('WSSinit')

    def initialize_xdmf_files(self):
        info('  Initializing output files.')
        # for creating paraview scripts
        self.metadata['filename_base'] = self.problem_code + '_' + self.metadata['name']

        # assemble file dictionary
        if self.doSaveDiff:
            self.fileDict.update(self.fileDictDiff)
        if self.metadata['hasTentativeV']:
            self.fileDict.update(self.fileDictTent)
            if self.doSaveDiff:
                self.fileDict.update(self.fileDictTentDiff)
        if self.metadata['hasTentativeP']:
            self.fileDict.update(self.fileDictTentP)
            if self.doSaveDiff:
                self.fileDict.update(self.fileDictTentPDiff)
        if self.args.ldsg:
            self.fileDict.update(self.fileDictLDSG)
        if self.args.wss != 'none':
            self.fileDict.update(self.fileDictWSSnorm)
            if self.args.wss_method == 'expression':
                self.fileDict.update(self.fileDictWSS)
        # create files
        for key, value in self.fileDict.iteritems():
            value['file'] = XDMFFile(mpi_comm_world(), self.str_dir_name + "/" + self.problem_code + '_' +
                                     self.metadata['name'] + value['name'] + ".xdmf")
            value['file'].parameters['rewrite_function_mesh'] = False  # saves lots of space (for use with static mesh)

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, is_tent, field):
        self.tc.start('div')
        self.divFunction.assign(project(div(field), self.divSpace))
        self.fileDict['d2' if is_tent else 'd']['file'] << (self.divFunction, self.actual_time)
        self.tc.end('div')

    def compute_div(self, is_tent, velocity):
        self.tc.start('divNorm')
        div_list = self.listDict['d2' if is_tent else 'd']['list']
        div_list.append(norm(velocity, 'Hdiv0'))
        if self.isWholeSecond:
            self.listDict['d2' if is_tent else 'd']['slist'].append(
                sum([i*i for i in div_list[self.N0:self.N1]])/self.stepsInSecond)
        self.tc.end('divNorm')

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field):
        self.vFunction.assign(field)
        self.fileDict['u2' if is_tent else 'u']['file'] << (self.vFunction, self.actual_time)
        if self.doSaveDiff:
            self.vFunction.assign((1.0 / self.vel_normalization_factor[0]) * (field - self.solution))
            self.fileDict['u2D' if is_tent else 'uD']['file'] << (self.vFunction, self.actual_time)
        if self.args.ldsg:
            # info(div(2.*sym(grad(field))-grad(field)).ufl_shape)
            form = div(2.*sym(grad(field))-grad(field))
            self.pFunction.assign(project(sqrt_ufl(inner(form, form)), self.pSpace))
            self.fileDict['ldsg2' if is_tent else 'ldsg']['file'] << (self.pFunction, self.actual_time)
            # self.vFunction.assign(project(div(2.*sym(grad(field))-grad(field)), self.vSpace))
            # self.fileDict['ldsg2' if is_tent else 'ldsg']['file'] << (self.vFunction, self.actual_time)

    def compute_err(self, is_tent, velocity, t):
        if self.doErrControl and self.has_analytic_solution:
            er_list_L2 = self.listDict['u2L2' if is_tent else 'u_L2']['list']
            er_list_H1 = self.listDict['u2H1' if is_tent else 'u_H1']['list']
            self.tc.start('errorV')
            errorL2_sq = assemble(inner(velocity - self.solution, velocity - self.solution) * dx)  # faster than errornorm
            errorH1seminorm_sq = assemble(inner(grad(velocity - self.solution), grad(velocity - self.solution)) * dx)  # faster than errornorm
            info('  H1 seminorm error: %f' % sqrt(errorH1seminorm_sq))
            errorL2 = sqrt(errorL2_sq)
            errorH1 = sqrt(errorL2_sq + errorH1seminorm_sq)
            info("  Relative L2 error in velocity = %f" % (errorL2 / self.analytic_v_norm_L2))
            self.last_error = errorH1 / self.analytic_v_norm_H1
            self.last_status_functional = self.last_error
            info("  Relative H1 error in velocity = %f" % self.last_error)
            er_list_L2.append(errorL2)
            er_list_H1.append(errorH1)
            self.tc.end('errorV')
            if self.testErrControl:
                er_list_test_H1 = self.listDict['u2H1test' if is_tent else 'u_H1test']['list']
                er_list_test_L2 = self.listDict['u2L2test' if is_tent else 'u_L2test']['list']
                self.tc.start('errorVtest')
                er_list_test_L2.append(errornorm(velocity, self.solution, norm_type='L2', degree_rise=0))
                er_list_test_H1.append(errornorm(velocity, self.solution, norm_type='H1', degree_rise=0))
                self.tc.end('errorVtest')
            if self.isWholeSecond:
                self.listDict['u2L2' if is_tent else 'u_L2']['slist'].append(
                    sqrt(sum([i*i for i in er_list_L2[self.N0:self.N1]])/self.stepsInSecond))
                self.listDict['u2H1' if is_tent else 'u_H1']['slist'].append(
                    sqrt(sum([i*i for i in er_list_H1[self.N0:self.N1]])/self.stepsInSecond))
            # stopping criteria
            if self.last_error > self.divergence_treshold:
                raise RuntimeError('STOPPED: Failed divergence test!')

    def averaging_pressure(self, pressure):
        self.tc.start('averageP')
        # averaging pressure (substract average)
        p_average = assemble((1.0/self.volume) * pressure * dx)
        info('Average pressure: %f' % p_average)
        p_average_function = interpolate(Expression("p", p=p_average), self.pSpace)
        # info(p_average_function, pressure, pressure_Q)
        pressure.assign(pressure - p_average_function)
        self.tc.end('averageP')

    def save_pressure(self, is_tent, pressure):
        self.tc.start('saveP')
        self.fileDict['p2' if is_tent else 'p']['file'] << (pressure, self.actual_time)
        # pg = project((1.0 / self.pg_normalization_factor[0]) * grad(pressure), self.pgSpace)  # NT normalisation factor defined only in Womersley
        # self.pgFunction.assign(pg)
        # self.fileDict['pg2' if is_tent else 'pg'][0] << (self.pgFunction, self.actual_time
        self.tc.end('saveP')

    def get_boundary_conditions(self, use_pressure_BC, v_space, p_space):
        pass

    def get_initial_conditions(self, function_list):
        """
        :param function_list: [{'type': 'v'/'p', 'time':-0.1},...]
        :return: velocities and pressures in selected times
        """
        pass

    def get_v_solution(self, t):
        pass

    def get_p_solution(self, t):
        pass

    def update_time(self, actual_time, step_number):
        info('t = %f, step %d' % (actual_time, step_number))
        self.actual_time = actual_time
        self.step_number = step_number
        self.time_list.append(self.actual_time)
        if self.onset < 0.001 or self.actual_time > self.onset:
            self.onset_factor = 1.
        else:
            self.onset_factor = (1. - cos(pi * actual_time / self.onset))*0.5
        info('Onset factor: %f' % self.onset_factor)

        # Manage saving choices for this step
        # save only n-th step in first second
        if self.doSave:
            dec, i = modf(actual_time)
            i = int(i)
            if self.args.ST == 'min':
                self.save_this_step = (i >= 1 and (step_number % self.stepsInSecond in self.peak_time_steps))
            elif self.args.ST == 'peak':
                self.save_this_step = (i >= 1 and 0.1 < dec < 0.20001)
            elif self.save_nth == 1 or i >= 1 or self.step_number % self.save_nth == 0:
                self.save_this_step = True
            else:
                self.save_this_step = False
            if self.save_this_step:
                info('Chose to save this step: (%f, %d)' % (actual_time, step_number))

    # this generates vector expression of normal on boundary mesh
    # for right orientation use BoundaryMesh(..., order=False)
    # IMP this does not work if some parallel process have no boundary mesh, because that is not implemented in FEniCS
    #
    class NormalExpression(Expression):
        """ This generates vector Expression of normal on boundary mesh.
        For right outward orientation use BoundaryMesh(mesh, 'exterior', order=False)  """
        def __init__(self, mesh):
            self.mesh = mesh
            self.gdim = mesh.geometry().dim()

        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            v=cell.get_vertex_coordinates().reshape((-1, self.gdim))
            vec1 = v[1] - v[0]   # create vectors by substracting coordinates of vertices
            vec2 = v[2] - v[0]
            n = np.cross(vec1, vec2)
            n /= np.linalg.norm(n)
            values[0] = n[0]
            values[1] = n[1]
            values[2] = n[2]

        def value_shape(self):
            return (3,)

    def compute_functionals(self, velocity, pressure, t, step):
        if self.args.wss == 'all' or (step >= self.stepsInSecond and self.args.wss == 'peak' and ((step % self.stepsInSecond) in self.peak_time_steps)):
            self.tc.start('WSS')
            begin('WSS (%dth step)' % step)
            if self.args.wss_method == 'expression':
                stress = project(self.nu*2*sym(grad(velocity)), self.T)
                stress.set_allow_extrapolation(True)
                stress_b = interpolate(stress, self.Tb)
                # self.fileDict['stress']['file'] << (stress_b, self.actual_time)
                info('Saved stress tensor')
                info('Computing WSS')
                wss = dot(stress_b, self.nb) - inner(dot(stress_b, self.nb), self.nb)*self.nb
                wss_func = project(wss, self.Vb)
                wss_norm = project(sqrt_ufl(inner(wss, wss)), self.Sb)
                info('Saving WSS')
                self.fileDict['wss']['file'] << (wss_func, self.actual_time)
                self.fileDict['wss_norm']['file'] << (wss_norm, self.actual_time)
            if self.args.wss_method == 'integral':
                wss_norm = Function(self.SDG)
                mS = TestFunction(self.SDG)
                scaling = 1/FacetArea(self.mesh)
                stress = self.nu*2*sym(grad(velocity))
                wss = dot(stress, self.normal) - inner(dot(stress, self.normal), self.normal)*self.normal
                wss_norm_form = scaling*mS*sqrt_ufl(inner(wss, wss))*ds
                assemble(wss_norm_form, tensor=wss_norm.vector())
                self.fileDict['wss_norm']['file'] << (wss_norm, self.actual_time)

                # NT this works, but it is hard to display in Paraview (DG, 1) vector space across exterior facet centers
                # wss_vector = []
                # for i in range(3):
                #     wss_component = Function(self.SDG)
                #     wss_vector_form = scaling*wss[i]*mS*ds
                #     assemble(wss_vector_form, tensor=wss_component.vector())
                #     wss_vector.append(wss_component)
                # wss_func = project(as_vector(wss_vector), self.VDG)
                # self.fileDict['wss']['file'] << (wss_func, self.actual_time)
            self.tc.end('WSS')
            end()

        # following was used to test laplace and stress formulation differences
        # dsgml = sqrt(assemble((1./self.mesh_volume)*inner(div(2*sym(grad(velocity))-grad(velocity)), div(2*sym(grad(velocity))-grad(velocity)))*dx))
        # dgt = sqrt(assemble((1./self.mesh_volume)*inner(div(transpose(grad(velocity))), div(transpose(grad(velocity))))*dx))
        # self.listDict['dsg-l']['list'].append(dsgml)
        # self.listDict['dgt']['list'].append(dgt)

    def compute_outflow(self, velocity):
        out = assemble(inner(velocity, self.normal)*self.get_outflow_measure_form())
        return out

    def get_outflow_measures(self):
        pass

    def get_outflow_measure_form(self):
        pass

    def get_metadata_to_save(self):
        return str(cPickle.dumps(self.metadata)).replace('\n', '$')

    def report(self):
        total = toc()
        md = self.metadata

        # compare errors measured by assemble and errornorm
        # TODO implement generally (if listDict[fcional]['testable'])
        # if self.testErrControl:
        #     for e in [[self.listDict['u_L2']['list'], self.listDict['u_L2test']['list'], 'L2'],
        #               [self.listDict['u_H1']['list'], self.listDict['u_H1test']['list'], 'H1']]:
        #         print('test ', e[2], sum([abs(e[0][i]-e[1][i]) for i in range(len(self.time_list))]))

        # report error norm, norm of div, and pressure gradients for individual time steps
        with open(self.str_dir_name + "/report_time_lines.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='\\', quoting=csv.QUOTE_NONE)
            # report_writer.writerow(self.problem.get_metadata_to_save())
            report_writer.writerow(["name", "what", "time"] + self.time_list)
            for key in self.listDict:
                l = self.listDict[key]
                if l['list']:
                    abrev = l['abrev']
                    report_writer.writerow([md['name'], l['name'], abrev] + l['list'])
                    if 'scale' in l:
                        temp_list = [i/l['scale'][0] for i in l['list']]
                        report_writer.writerow([md['name'], "scaled " + l['name'], abrev+"s"] + temp_list +
                                               ['scale factor:' + str(l['scale'])])
                    if 'norm' in l:
                        if l['norm']:
                            temp_list = [i/l['norm'][0] for i in l['list']]
                            report_writer.writerow([md['name'], "normalized " + l['name'], abrev+"n"] + temp_list)
                        else:
                            info('Norm missing:' + str(l))
                            l['normalized_list_sec'] = []
                    if 'relative' in l:
                        norm_list = self.listDict[l['relative']]['list']
                        temp_list = [l['list'][i]/norm_list[i] for i in range(0, len(l['list']))]
                        self.listDict[key]['relative_list'] = temp_list
                        report_writer.writerow([md['name'], "relative " + l['name'], abrev+"r"] + temp_list)

        # report error norm, norm of div, and pressure gradients averaged over seconds
        with open(self.str_dir_name + "/report_seconds.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='|', quoting=csv.QUOTE_NONE)
            # report_writer.writerow(self.problem.get_metadata_to_save())
            report_writer.writerow(["name", "what", "time"] + self.second_list)
            for key in self.listDict.iterkeys():
                l = self.listDict[key]
                if 'slist' in l:
                    abrev = l['abrev']
                    value = l['slist']
                    report_writer.writerow([md['name'], l['name'], abrev] + value)
                    if 'scale' in l:
                        temp_list = [i/l['scale'][0] for i in value]
                        report_writer.writerow([md['name'], "scaled " + l['name'], abrev+"s"] + temp_list)
                    if 'norm' in l:
                        if l['norm']:
                            temp_list = [i/l['norm'][0] for i in value]
                            l['normalized_list_sec'] = temp_list
                            report_writer.writerow([md['name'], "normalized " + l['name'], abrev+"n"] + temp_list)
                        else:
                            info('Norm missing:' + str(l))
                            l['normalized_list_sec'] = []
                    if 'relative_list' in l:
                        temp_list = []
                        # info('relative second list of'+ str(l['abrev']))
                        for sec in self.second_list:
                            N0 = (sec-1)*self.stepsInSecond
                            N1 = sec*self.stepsInSecond
                            # info([sec,  N0, N1])
                            temp_list.append(sqrt(sum([i*i for i in l['relative_list'][N0:N1]])/float(self.stepsInSecond)))
                        l['relative_list_sec'] = temp_list
                        report_writer.writerow([md['name'], "relative " + l['name'], abrev+"r"] + temp_list)

        header_row = ["name", 'metadata', "totalTimeHours"]
        data_row = [md['name'], self.get_metadata_to_save(), total / 3600.0]
        for key in ['u_L2', 'u_H1', 'u_H1w', 'p', 'u2L2', 'u2H1', 'u2H1w', 'p2', 'pgE', 'pgE2', 'd', 'd2', 'force_wall']:
            if key in self.listDict:
                l = self.listDict[key]
                header_row += ['last_cycle_'+l['abrev']]
                data_row += [l['slist'][-1]] if l['slist'] else [0]
                if 'relative_list_sec' in l and l['relative_list_sec']:
                    header_row += ['last_cycle_'+l['abrev']+'r']
                    data_row += [l['relative_list_sec'][-1]]
                elif key in ['p', 'p2']:
                    header_row += ['last_cycle_'+l['abrev']+'n']
                    data_row += [l['normalized_list_sec'][-1]] if l['normalized_list_sec'] else [0]

        # report without header
        with open(self.str_dir_name + "/report.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(data_row)

        # report with header
        with open(self.str_dir_name + "/report_h.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(header_row)
            report_writer.writerow(data_row)

        # report time cotrol
        with open(self.str_dir_name + "/report_timecontrol.csv", 'w') as reportFile:
            self.tc.report(reportFile, self.metadata['name'])

        self.remove_status_file()

        # create file showing all was done well
        f = open(md['name'] + "_OK.report", "w")
        f.close()

    def report_fail(self, t):
        print("Runtime error:", sys.exc_info()[1])
        print("Traceback:")
        traceback.print_tb(sys.exc_info()[2])
        f = open(self.metadata['name'] + "_failed_at_%5.3f.report" % t, "w")
        f.write(traceback.format_exc())
        f.close()
        self.remove_status_file()

    def write_status_file(self, t):
        self.tc.start('status')
        f = open(self.metadata['name'] + ".run", "w")
        progress = t/self.metadata['time']
        f.write('t = %5.3f (dt=%3dms)\nprogress = %3.0f %%\n%s = %5.3f\n' %
                (t, self.metadata['dt_ms'], 100*progress, self.status_functional_str, self.last_status_functional))
        f.close()
        self.tc.end('status')

    def remove_status_file(self):
        if self.MPI_rank == 0:
            try:
                os.remove(self.metadata['name'] + ".run")
            except OSError:
                info('.run file probably not created')

