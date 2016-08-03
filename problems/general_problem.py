from __future__ import print_function

import cPickle
import csv
import numpy as np
import os
import sys
import traceback
from dolfin import Function, TestFunction, assemble, interpolate, Expression, project, norm, errornorm, \
    TensorFunctionSpace, FunctionSpace, VectorFunctionSpace
from dolfin.cpp.common import mpi_comm_world, toc, MPI, info, begin, end
from dolfin.cpp.io import XDMFFile, HDF5File
from dolfin.cpp.mesh import Mesh, MeshFunction, BoundaryMesh, Cell
from math import sqrt, pi, cos, modf
from ufl import dx, div, inner, grad, sym, sqrt as sqrt_ufl, dot, FacetArea, ds


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
        # Time control
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
        self.tc.init_watch('errorV', 'Computed velocity error', True)
        self.tc.init_watch('errorVtest', 'Computed velocity error test', True)

        # If it is sensible (and implemented) to force pressure gradient on outflow boundary
        # 1. set self.outflow_area in initialize
        # 2. implement self.compute_outflow and get_outflow_measures
        self.can_force_outflow = False
        self.outflow_area = None
        self.outflow_measures = []

        # stopping criteria (for relative H1 velocity error norm) (if known)
        self.divergence_treshold = 10

        # used for writing status files .run to monitor progress during computation:
        self.last_status_functional = 0.0
        self.status_functional_str = 'to be defined in Problem class'

        # mesh and function objects
        self.normal = None
        self.mesh = None
        self.facet_function = None
        self.mesh_volume = 0   # must be specified in subclass (needed to compute pressure average)
        self.stepsInSecond = None
        self.vSpace = None
        self.vFunction = None
        self.divSpace = None
        self.divFunction = None
        self.pSpace = None
        # self.pgSpace = None    # NT computing pressure gradient function not used (commented throughout code)
        self.pFunction = None
        # self.pgFunction = None
        self.solutionSpace = None
        self.solution = None

        # time related variables
        self.actual_time = 0.0
        self.step_number = 0
        self.save_this_step = False
        self.isWholeSecond = None
        self.N1 = None
        self.N0 = None

        # lists used for storing normalization and scaling coefficients
        # (time independent, mainly used in womersley_cylinder)
        self.vel_normalization_factor = []
        self.pg_normalization_factor = []
        self.p_normalization_factor = []
        self.scale_factor = []

        self.analytic_v_norm_L2 = None
        self.analytic_v_norm_H1 = None
        self.analytic_v_norm_H1w = None
        self.last_error = None

        # for WSS generation
        metadata['hasWSS'] = (args.wss != 'none')
        metadata['WSSmethod'] = self.args.wss_method
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

        # dictionaries for output XDMF files
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
        self.fileDictWSS = {'wss': {'name': 'wss'}, }
        self.fileDictWSSnorm = {'wss_norm': {'name': 'wss_norm'}, }
        self.fileDictDebugRot = {'grad_cor': {'name': 'grad_cor'}, }

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
            'd': {'list': [], 'name': 'corrected velocity L2 divergence', 'abrev': 'DC',
                  'scale': self.scale_factor, 'slist': []},
            'd2': {'list': [], 'name': 'tentative velocity L2 divergence', 'abrev': 'DT',
                   'scale': self.scale_factor, 'slist': []},
            'pg': {'list': [], 'name': 'computed pressure gradient', 'abrev': 'PG', 'scale': self.scale_factor,
                   'norm': self.pg_normalization_factor},
            'pg2': {'list': [], 'name': 'computed pressure tent gradient', 'abrev': 'PTG', 'scale': self.scale_factor,
                    'norm': self.pg_normalization_factor},
        }
        if self.has_analytic_solution:
            self.listDict.update({
                'u_L2': {'list': [], 'name': 'corrected velocity L2 error', 'abrev': 'CE_L2',
                         'scale': self.scale_factor, 'relative': 'av_norm_L2', 'slist': [],
                         'norm': self.vel_normalization_factor},
                'u2L2': {'list': [], 'name': 'tentative velocity L2 error', 'abrev': 'TE_L2',
                         'scale': self.scale_factor, 'relative': 'av_norm_L2', 'slist': [],
                         'norm': self.vel_normalization_factor},
                'u_L2test': {'list': [], 'name': 'test corrected L2 velocity error', 'abrev': 'TestCE_L2',
                             'scale': self.scale_factor},
                'u2L2test': {'list': [], 'name': 'test tentative L2 velocity error', 'abrev': 'TestTE_L2',
                             'scale': self.scale_factor},
                'u_H1': {'list': [], 'name': 'corrected velocity H1 error', 'abrev': 'CE_H1',
                         'scale': self.scale_factor, 'relative': 'av_norm_H1', 'slist': []},
                'u2H1': {'list': [], 'name': 'tentative velocity H1 error', 'abrev': 'TE_H1',
                         'scale': self.scale_factor, 'relative': 'av_norm_H1', 'slist': []},
                'u_H1test': {'list': [], 'name': 'test corrected H1 velocity error', 'abrev': 'TestCE_H1',
                             'scale': self.scale_factor},
                'u2H1test': {'list': [], 'name': 'test tentative H1 velocity error', 'abrev': 'TestTE_H1',
                             'scale': self.scale_factor},
                'apg': {'list': [], 'name': 'analytic pressure gradient', 'abrev': 'APG', 'scale': self.scale_factor,
                        'norm': self.pg_normalization_factor},
                'av_norm_L2': {'list': [], 'name': 'analytic velocity L2 norm', 'abrev': 'AVN_L2'},
                'av_norm_H1': {'list': [], 'name': 'analytic velocity H1 norm', 'abrev': 'AVN_H1'},
                'ap_norm': {'list': [], 'name': 'analytic pressure norm', 'abrev': 'APN'},
                'p': {'list': [], 'name': 'pressure L2(0) error', 'abrev': 'PE', 'scale': self.scale_factor,
                      'slist': [], 'norm': self.p_normalization_factor},
                'pgE': {'list': [], 'name': 'computed pressure gradient error', 'abrev': 'PGE',
                        'scale': self.scale_factor, 'norm': self.pg_normalization_factor, 'slist': []},
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
        self.nu = 0.0  # nu should be specified in subclass (fixed or by argument)
        self.onset = args.onset
        self.onset_factor = 0
        # saving options:
        self.doSave = False
        self.saveOnlyVel = False
        self.doSaveDiff = False
        self.save_nth = args.saventh
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
        # error control options:
        self.doErrControl = None
        self.testErrControl = False
        if not self.has_analytic_solution:
            self.doErrControl = False
        elif args.error == "noEC":
            self.doErrControl = False
            info("Error control OFF")
        else:
            self.doErrControl = True
            if args.error == "test":
                self.testErrControl = True
                info("Error control in testing mode")
            else:
                info("Error control ON")

        # set directory for results and reports
        self.str_dir_name = "%s_%s_results" % (self.problem_code, metadata['name'])
        self.metadata['dir'] = self.str_dir_name
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name) and self.MPI_rank == 0:
            os.mkdir(self.str_dir_name)

    @staticmethod
    def setup_parser_options(parser):
        parser.add_argument('-S', '--save', help='Save solution mode', choices=['doSave', 'noSave', 'diff', 'only_vel'],
                            default='noSave')
        #   doSave: create .xdmf files with velocity, pressure, divergence
        #   diff: save also difference vel-sol
        #   noSave: do not create .xdmf files with velocity, pressure, divergence
        parser.add_argument('--saventh', help='save only n-th step in first cycle', type=int, default=1)
        parser.add_argument('--ST', help='save only n-th step in first cycle',
                            choices=['min', 'peak', 'no_restriction'], default='no_restriction')
        # stronger than --saventh, to be used instead of it
        parser.add_argument('--onset', help='boundary condition onset length', type=float, default=0.5)
        parser.add_argument('--wss', help='compute wall shear stress', choices=['none', 'all', 'peak'], default='none')
        # --wss is independent of -S options
        parser.add_argument('--wss_method', help='compute wall shear stress', choices=['expression', 'integral'],
                            default='integral')
        # 'expression' works with BoundaryMesh object. It restricts stress to boundary mesh and then computes CG,1
        # vector WSS and its magnitude from values on boundary
        # expression does not work for too many processors (24 procs for 'HYK' is OK, 48 is too much), because the case
        # when some parallel process have no boundary mesh is not implemented in FEniCS
        # 'integral' works always. It computes facet average wss norm using assemble() to integrate over boundary
        #  facets. Results for each cell are stored in DG,0 scalar function (interior cells will have value 0, because
        # only exterior facet integrals are computed)
        # Note: it does not make sense to project the result into any other space, because zeros inside domain would
        # interfere with values on boundary
        parser.add_argument('--debug_rot', help='save more information about rotation correction', action='store_true')
        # idea was to save rotational correction contribution to velocity RHS
        # NT results are not reliable. One would need to project nu*div(v) into Q space before computing gradient.
        # NT not documented in readme
        parser.add_argument('-e', '--error', help='Error control mode', choices=['doEC', 'noEC', 'test'],
                            default='doEC')
        # compute error using analytic solution (if available)
        # 'test' mode computes errors using both assemble() and errornorm() to check if the results are equal

    @staticmethod
    def loadMesh(mesh):
        """
        :param mesh: name of mesh file (without extension)
        :return: tuple mesh, facet function read from .hdf5 file
        """
        f = HDF5File(mpi_comm_world(), 'meshes/'+mesh+'.hdf5', 'r')
        mesh = Mesh()
        f.read(mesh, 'mesh', False)
        facet_function = MeshFunction("size_t", mesh)
        f.read(facet_function, 'facet_function')
        return mesh, facet_function

    def initialize(self, V, Q, PS, D):
        """
        :param V: velocity space
        :param Q: pressure space
        :param PS: scalar space of same order as V, used for analytic solution generation
        :param D: divergence of velocity space
        """
        self.vSpace = V
        self.divSpace = D
        self.pSpace = Q
        self.solutionSpace = V
        self.vFunction = Function(V)
        self.divFunction = Function(D)
        self.pFunction = Function(Q)

        # self.pgSpace = VectorFunctionSpace(mesh, "DG", 0)    # used to save pressure gradient as vectors
        # self.pgFunction = Function(self.pgSpace)
        self.initialize_xdmf_files()
        self.stepsInSecond = int(round(1.0 / self.metadata['dt']))
        info('stepsInSecond = %d' % self.stepsInSecond)

        if self.args.ST == 'min' or self.args.wss == 'peak':
            # 0.166... is peak for real problem, 0.188 is peak for womersley profile
            chosen_steps = [0.1, 0.125, 0.15, 0.16, 0.165, 0.166, 0.167, 0.17, 0.188, 0.189, 0.2]
            # select computed steps nearest to chosen steps:
            self.peak_time_steps = [int(round(chosen / self.metadata['dt'])) for chosen in chosen_steps]
            for ch in self.peak_time_steps:
                info('Chosen peak time steps at every %dth step in %d steps' % (ch, self.stepsInSecond))

        if self.args.wss != 'none':
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
        if self.doSave:
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
        else:
            self.fileDict = {}
        if self.args.wss != 'none':
            self.fileDict.update(self.fileDictWSSnorm)
            if self.args.wss_method == 'expression':
                self.fileDict.update(self.fileDictWSS)
        if self.args.debug_rot:
            self.fileDict.update(self.fileDictDebugRot)
        # create and setup files
        for key, value in self.fileDict.iteritems():
            value['file'] = XDMFFile(mpi_comm_world(), self.str_dir_name + "/" + self.problem_code + '_' +
                                     self.metadata['name'] + value['name'] + ".xdmf")
            value['file'].parameters['rewrite_function_mesh'] = False  # save mesh only once per file

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
        self.tc.end('divNorm')

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field):
        self.vFunction.assign(field)
        self.fileDict['u2' if is_tent else 'u']['file'] << (self.vFunction, self.actual_time)
        if self.doSaveDiff:
            self.vFunction.assign((1.0 / self.vel_normalization_factor[0]) * (field - self.solution))
            self.fileDict['u2D' if is_tent else 'uD']['file'] << (self.vFunction, self.actual_time)

    def compute_err(self, is_tent, velocity, t):
        if self.doErrControl:
            er_list_L2 = self.listDict['u2L2' if is_tent else 'u_L2']['list']
            er_list_H1 = self.listDict['u2H1' if is_tent else 'u_H1']['list']
            self.tc.start('errorV')
            # assemble is faster than errornorm
            errorL2_sq = assemble(inner(velocity - self.solution, velocity - self.solution) * dx)
            errorH1seminorm_sq = assemble(inner(grad(velocity - self.solution), grad(velocity - self.solution)) * dx)
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
            # stopping criteria for detecting diverging solution
            if self.last_error > self.divergence_treshold:
                raise RuntimeError('STOPPED: Failed divergence test!')

    def averaging_pressure(self, pressure):
        """
        :param pressure: average is subtracted from it
        """
        self.tc.start('averageP')
        # averaging pressure (subtract average)
        p_average = assemble((1.0 / self.mesh_volume) * pressure * dx)
        info('Average pressure: %f' % p_average)
        p_average_function = interpolate(Expression("p", p=p_average), self.pSpace)
        pressure.assign(pressure - p_average_function)
        self.tc.end('averageP')

    def save_pressure(self, is_tent, pressure):
        self.tc.start('saveP')
        self.fileDict['p2' if is_tent else 'p']['file'] << (pressure, self.actual_time)
        # NT normalisation factor defined only in Womersley
        # pg = project((1.0 / self.pg_normalization_factor[0]) * grad(pressure), self.pgSpace)
        # self.pgFunction.assign(pg)
        # self.fileDict['pg2' if is_tent else 'pg'][0] << (self.pgFunction, self.actual_time
        self.tc.end('saveP')

    def get_boundary_conditions(self, use_pressure_BC, v_space, p_space):
        info('get_boundary_conditions() for this problem was not properly implemented.')

    def get_initial_conditions(self, function_list):
        """
        :param function_list: [{'type': 'v'/'p', 'time':-0.1},...]
        :return: velocities and pressures in selected times
        """
        info('get_initial_conditions for this problem was not properly implemented.')

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
    class NormalExpression(Expression):
        """ This generates vector Expression of normal on boundary mesh.
        For right outward orientation use BoundaryMesh(mesh, 'exterior', order=False)  """
        def __init__(self, mesh):
            self.mesh = mesh
            self.gdim = mesh.geometry().dim()

        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            v=cell.get_vertex_coordinates().reshape((-1, self.gdim))
            vec1 = v[1] - v[0]   # create vectors by subtracting coordinates of vertices
            vec2 = v[2] - v[0]
            n = np.cross(vec1, vec2)
            n /= np.linalg.norm(n)
            values[0] = n[0]
            values[1] = n[1]
            values[2] = n[2]

        def value_shape(self):
            return 3,

    def compute_functionals(self, velocity, pressure, t, step):
        if self.args.wss == 'all' or \
                (step >= self.stepsInSecond and self.args.wss == 'peak' and
                 ((step % self.stepsInSecond) in self.peak_time_steps)):
            self.tc.start('WSS')
            begin('WSS (%dth step)' % step)
            if self.args.wss_method == 'expression':
                stress = project(self.nu*2*sym(grad(velocity)), self.T)
                # pressure is not used as it contributes only to the normal component
                stress.set_allow_extrapolation(True)   # need because of some inaccuracies in BoundaryMesh coordinates
                stress_b = interpolate(stress, self.Tb)    # restrict stress to boundary mesh
                # self.fileDict['stress']['file'] << (stress_b, self.actual_time)
                # info('Saved stress tensor')
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
                wss_norm_form = scaling*mS*sqrt_ufl(inner(wss, wss))*ds   # ds is integral over exterior facets only
                assemble(wss_norm_form, tensor=wss_norm.vector())
                self.fileDict['wss_norm']['file'] << (wss_norm, self.actual_time)

                # to get vector WSS values:
                # NT this works, but in ParaView for (DG,1)-vector space glyphs are displayed in cell centers
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

    def compute_outflow(self, velocity):
        out = assemble(inner(velocity, self.normal)*self.get_outflow_measure_form())
        return out

    def get_outflow_measures(self):
        """
        return list of Measure objects for outflow boundary parts
        must be overridden in Problem before use (if needed)
        """
        info('Integration over outflow for this problem was not properly implemented.')

    def get_outflow_measure_form(self):
        """
        returns term (dSout1 + ... + dSoutN) that can be used in form to integrate over all outflow boundary parts
        works if get_outflow_measures is implemented
        """
        if not self.outflow_measures:
            self.outflow_measures = self.get_outflow_measures()
        if len(self.outflow_measures) == 1:
            return self.outflow_measures[0]
        else:
            out = self.outflow_measures[0]
            for m in self.outflow_measures[1:]:
                out += m
            return out

    def get_metadata_to_save(self):
        return str(cPickle.dumps(self.metadata))#.replace('\n', '$')

    # noinspection PyTypeChecker
    def report(self):
        total = toc()
        md = self.metadata

        # compare errors measured by assemble and errornorm
        # TODO implement generally (if listDict[fcional]['testable'])
        # if self.testErrControl:
        #     for e in [[self.listDict['u_L2']['list'], self.listDict['u_L2test']['list'], 'L2'],
        #               [self.listDict['u_H1']['list'], self.listDict['u_H1test']['list'], 'H1']]:
        #         print('test ', e[2], sum([abs(e[0][i]-e[1][i]) for i in range(len(self.time_list))]))

        # report error norms, norms of divergence etc. for individual time steps
        with open(self.str_dir_name + "/report_time_lines.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='\\', quoting=csv.QUOTE_NONE)
            # report_writer.writerow(self.problem.get_metadata_to_save())
            report_writer.writerow(["name", "what", "time"] + self.time_list)
            keys = sorted(self.listDict.keys())
            for key in keys:
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

        # report error norms, norms of divergence etc. averaged over seconds
        if self.second_list:
            with open(self.str_dir_name + "/report_seconds.csv", 'w') as reportFile:
                report_writer = csv.writer(reportFile, delimiter=';', escapechar='|', quoting=csv.QUOTE_NONE)
                report_writer.writerow(["name", "what", "time"] + self.second_list)
                keys = sorted(self.listDict.keys())
                for key in keys:
                    l = self.listDict[key]
                    if 'slist' in l and l['list']:
                        abrev = l['abrev']
                        values = l['slist']
                        # generate averages over seconds from list
                        for sec in self.second_list:
                            N0 = (sec-1)*self.stepsInSecond
                            N1 = sec*self.stepsInSecond
                            values.append(sqrt(sum([i*i for i in l['list'][N0:N1]])/float(self.stepsInSecond)))

                        report_writer.writerow([md['name'], l['name'], abrev] + values)
                        if 'scale' in l:
                            temp_list = [i/l['scale'][0] for i in values]
                            report_writer.writerow([md['name'], "scaled " + l['name'], abrev+"s"] + temp_list)
                        if 'norm' in l:
                            if l['norm']:
                                temp_list = [i/l['norm'][0] for i in values]
                                l['normalized_list_sec'] = temp_list
                                report_writer.writerow([md['name'], "normalized " + l['name'], abrev+"n"] + temp_list)
                            else:
                                info('Norm missing:' + str(l))
                                l['normalized_list_sec'] = []
                        if 'relative_list' in l:
                            temp_list = []
                            for sec in self.second_list:
                                N0 = (sec-1)*self.stepsInSecond
                                N1 = sec*self.stepsInSecond
                                temp_list.append(sqrt(sum([i*i for i in l['relative_list'][N0:N1]])/float(self.stepsInSecond)))
                            l['relative_list_sec'] = temp_list
                            report_writer.writerow([md['name'], "relative " + l['name'], abrev+"r"] + temp_list)

        header_row = ["name", "totalTimeHours"]
        data_row = [md['name'], total / 3600.0]
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
                    data_row += [l['normalized_list_sec'][-1]] if 'normalized_list_sec' in l \
                                                                  and l['normalized_list_sec'] else [0]

        # save metadata. Can be loaded and used in postprocessing
        with open(self.str_dir_name + "/metadata", 'w') as reportFile:
            reportFile.write(self.get_metadata_to_save())

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
        f.write('t = %5.3f (dt=%f)\nprogress = %3.0f %%\n%s = %5.3f\n' %
                (t, self.metadata['dt'], 100*progress, self.status_functional_str, self.last_status_functional))
        f.close()
        self.tc.end('status')

    def remove_status_file(self):
        if self.MPI_rank == 0:
            try:
                os.remove(self.metadata['name'] + ".run")
            except OSError:
                info('.run file probably not created')

