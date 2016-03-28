from __future__ import print_function
import os
from dolfin import Function, assemble, interpolate, Expression
from dolfin.cpp.common import mpi_comm_world
from dolfin.cpp.io import XDMFFile
from ufl import dx

class GeneralProblem:
    def __init__(self, args, tc, metadata):
        self.tc = tc

        self.tc.init_watch('saveP', 'Saved pressure', True)
        self.tc.init_watch('saveVel', 'Saved velocity', True)
        self.tc.init_watch('averageP', 'Averaged pressure', False)   # falls within saveP

        self.args = args
        self.metadata = metadata

        # need to be specified in subclass constructor before calling this constructor
        # IMP this code looks awful: how to do it properly? Possibly move dependent steps to initialization?
        self.problem_code = self.problem_code
        self.has_analytic_solution = self.has_analytic_solution

        self.last_status_functional = 0.0
        self.status_functional_str = 'to be defined in Problem class'

        self.stepsInSecond = None
        self.volume = None
        self.vSpace = None
        self.vFunction = None
        self.pSpace = None
        self.pFunction = None
        self.solutionSpace = None
        self.partialSolutionSpace = None  # scalar space of same element type as velocity
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

        self.isWholeSecond = None
        self.N1 = None
        self.N0 = None

        # lists
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.second_list = []
        self.listDict = {}  # list of fuctionals

        # parse arguments
        self.nu_factor = args.nu

        self.doSave = False
        self.doSaveDiff = False
        option = args.save
        if option == 'doSave' or option == 'diff':
            self.doSave = True
            if option == 'diff':
                self.doSaveDiff = True
                print('Saving velocity differences.')
            print('Saving solution ON.')
        elif option == 'noSave':
            self.doSave = False
            print('Saving solution OFF.')

        self.doErrControl = None
        self.testErrControl = False
        if args.error == "noEC":
            self.doErrControl = False
            print("Error control omitted")
        else:
            self.doErrControl = True
            if args.error == "test":
                self.testErrControl = True
                print("Error control in testing mode")
            else:
                print("Error control on")

        self.str_dir_name = "%s_%s_results" % (self.problem_code, metadata['name'])
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name):
            os.mkdir(self.str_dir_name)


    @staticmethod
    def setup_parser_options(parser):
        parser.add_argument('-e', '--error', help='Error control mode', choices=['doEC', 'noEC', 'test'], default='doEC')
        parser.add_argument('-S', '--save', help='Save solution mode', choices=['doSave', 'noSave', 'diff'], default='noSave')
        #   doSave: create .xdmf files with velocity, pressure, divergence
        #   diff: save also difference vel-sol
        #   noSave: do not create .xdmf files with velocity, pressure, divergence
        parser.add_argument('--nu', help='kinematic viscosity factor', type=float, default=1.0)

    def initialize(self, V, Q, PS):
        self.vSpace = V
        self.pSpace = Q
        self.solutionSpace = V
        self.partialSolutionSpace = PS
        self.vFunction = Function(V)
        self.pFunction = Function(Q)
        self.volume = assemble(interpolate(Expression("1.0"), Q) * dx)

        if self.doSave:
            # self.D = FunctionSpace(mesh, "Lagrange", 1)
            # self.pgSpace = VectorFunctionSpace(mesh, "DG", 0)
            # self.pgFunction = Function(self.pgSpace)
            self.initialize_xdmf_files()
        self.stepsInSecond = int(round(1.0 / self.metadata['dt']))
        print('stepsInSecond = ', self.stepsInSecond)

    def initialize_xdmf_files(self):
        print('  Initializing output files.')
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
        # create files
        for key, value in self.fileDict.iteritems():
            value['file'] = XDMFFile(mpi_comm_world(), self.str_dir_name + "/" + value['name'] + ".xdmf")
            value['file'].parameters['rewrite_function_mesh'] = False  # saves lots of space (for use with static mesh)

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field, t):
        self.vFunction.assign(field)
        self.fileDict['u2' if is_tent else 'u']['file'] << self.vFunction

    def averaging_pressure(self, pressure):
        self.tc.start('averageP')
        # averaging pressure (substract average)
        p_average = assemble((1.0/self.volume) * pressure * dx)
        print('Average pressure: ', p_average)
        p_average_function = interpolate(Expression("p", p=p_average), self.pSpace)
        # print(p_average_function, pressure, pressure_Q)
        pressure.assign(pressure - p_average_function)
        self.tc.end('averageP')

    def save_pressure(self, is_tent, pressure):
        if self.doSave:
            self.fileDict['p2' if is_tent else 'p']['file'] << pressure
            # pg = project((1.0 / self.pg_normalization_factor[0]) * grad(pressure), self.pgSpace)
            # self.pgFunction.assign(pg)
            # self.fileDict['pg2' if is_tent else 'pg'][0] << self.pgFunction

    def get_boundary_conditions(self, *args):
        pass

    def get_initial_conditions(self, *args):
        pass

    def update_time(self, actual_time):
        pass

    def report(self):
        # report time cotrol
        with open(self.str_dir_name + "/report_timecontrol.csv", 'w') as reportFile:
            self.tc.report(reportFile, self.metadata['name'])
