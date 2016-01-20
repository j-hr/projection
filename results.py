from __future__ import print_function

__author__ = 'jh'

from dolfin import *
import os, traceback, math, csv, sys
import womersleyBC
import problem as prb


class TimeControl:
    def __init__(self):
        # watch is list [total_time, last_start, message_when_measured]
        self.watches = {}
        tic()

    def init_watch(self, what, message, count_to_sum):
        if what not in self.watches:
            self.watches[what] = [0, 0, message, count_to_sum]

    def start(self, what):
        if what in self.watches:
            self.watches[what][1] = toc()

    def end(self, what):
        watch = self.watches[what]
        elapsed = toc() - watch[1]
        watch[0] += elapsed
        print(watch[2]+'. Time: %.4f Total: %.4f' % (elapsed, watch[0]))

    def report(self, report_file, str_name):
        total_time = toc()
        print('Total time of %.0f s, (%.2f hours).' % (total_time, total_time/3600.0))
        sorted_by_time = []
        sorted_by_name = []
        sum = 0
        for value in self.watches.itervalues():
            if value[3]:
               sum += value[0]
            if not sorted_by_time:
                sorted_by_time.append(value)
            else:
                i = 0
                l = len(sorted_by_time)
                while i < l and value[0]<sorted_by_time[i][0]:
                    i += 1
                sorted_by_time.insert(i, value)
        for value in sorted_by_time:
            print('   %-40s: %12.2f s %5.1f %%' % (value[2], value[0], 100.0*value[0]/total_time))
        print('   %-40s: %12.2f s %5.1f %%' % ('Measured', sum, 100.0*sum/total_time))
        print('   %-40s: %12.2f s %5.1f %%' % ('Unmeasured', total_time-sum, 100.0*(total_time-sum)/total_time))
        # report to file
        for key in self.watches.iterkeys():   # sort keys by name
            if not sorted_by_name:
                sorted_by_name.append(key)
            else:
                l = len(sorted_by_name)
                i = l
                while key < sorted_by_name[i-1] and i > 0:
                    i -= 1
                sorted_by_name.insert(i, key)
        report_header = ['Name', 'Total time']
        report_data = [str_name, total_time]
        for key in sorted_by_name:
            value = self.watches[key]
            report_header.append(value[2])
            report_header.append('part '+value[2])
            report_data.append(value[0])
            report_data.append(value[0]/total_time)
        report_header.append('part unmeasured')
        report_data.append((total_time-sum)/total_time)
        writer = csv.writer(report_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
        writer.writerow(report_header)
        writer.writerow(report_data)


class ResultsManager:
    def __init__(self, problem, time_control):
        self.problem = problem

        # fixed parameters (used in analytic solution and in BC)
        self.nu = 3.71  # kinematic viscosity
        self.R = 5.0  # cylinder radius

        # from main
        self.factor = None
        self.tc = time_control
        self.dsWall = None
        self.tc.init_watch('assembleSol', 'Assembled analytic solution', True)
        self.tc.init_watch('analyticP', 'Analytic pressure', True)
        self.tc.init_watch('analyticVnorms', 'Computed analytic velocity norms', True)
        self.tc.init_watch('saveP', 'Saved pressure', True)
        self.tc.init_watch('errorP', 'Computed pressure error', True)
        self.tc.init_watch('errorV', 'Computed velocity error', True)
        self.tc.init_watch('errorForce', 'Computed force error', True)
        self.tc.init_watch('errorVtest', 'Computed velocity error test', True)
        self.tc.init_watch('div', 'Computed and saved divergence', True)
        self.tc.init_watch('divNorm', 'Computed norm of divergence', True)
        self.tc.init_watch('saveVel', 'Saved velocity', True)

        # partial Bessel functions and coefficients
        self.bessel_parabolic = None
        self.bessel_real = []
        self.bessel_complex = []
        self.coefs_exp = None

        self.str_dir_name = None
        self.doSave = False
        self.doSaveDiff = False
        option = self.problem.saveoption
        if option == 'save' or option == 'diff':
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
        self.isSteadyFlow = None

        self.vel_normalization_factor = []
        self.pg_normalization_factor = []
        self.p_normalization_factor = []
        self.solutionSpace = None
        self.solution = None
        self.sol_p = None
        self.time_erc = 0  # total time spent on measuring error
        self.actual_time = None
        self.isWholeSecond = False
        self.N0 = 0
        self.N1 = 0
        self.stepsInSecond = None
        self.analytic_v_norm_L2 = 0
        self.analytic_v_norm_H1 = 0
        self.analytic_v_norm_H1w = 0
        self.last_analytic_pressure_norm = 0
        self.vel = None
        self.D = None
        self.divFunction = None
        self.pSpace = None
        self.pFunction = None
        self.pgSpace = None
        self.pgFunction = None

        # lists
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.second_list = []
        # dictionary of data lists {list, name, abbreviation, add scaled row to report}
        # normalisation coefficients (time-independent) are added to some lists to be used in normalized data series
        #   coefs are equal to average of respective value of analytic solution
        # norm lists (time-dependent normalisation coefficients) are added to some lists to be used in relative data
        #  series (to remove natural pulsation of error due to change in volume flow rate)
        # slist - lists for cycle-averaged values
        # L2(0) means L2 difference of pressures taken with zero average
        self.listDict = {
            'u_L2': {'list': [], 'name': 'corrected velocity L2 error', 'abrev': 'CE_L2', 'scale': True,
                     'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
            'u2L2': {'list': [], 'name': 'tentative velocity L2 error', 'abrev': 'TE_L2', 'scale': True,
                     'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
            'u_L2test': {'list': [], 'name': 'test corrected L2 velocity error', 'abrev': 'TestCE_L2', 'scale': True},
            'u2L2test': {'list': [], 'name': 'test tentative L2 velocity error', 'abrev': 'TestTE_L2', 'scale': True},
            'u_H1': {'list': [], 'name': 'corrected velocity H1 error', 'abrev': 'CE_H1', 'scale': True,
                     'relative': 'av_norm_H1', 'slist': []},
            'u_H1w': {'list': [], 'name': 'corrected velocity H1 error on wall', 'abrev': 'CE_H1w', 'scale': True,
                     'relative': 'av_norm_H1w', 'slist': []},
            'u2H1': {'list': [], 'name': 'tentative velocity H1 error', 'abrev': 'TE_H1', 'scale': True,
                     'relative': 'av_norm_H1', 'slist': []},
            'u2H1w': {'list': [], 'name': 'tentative velocity H1 error on wall', 'abrev': 'TE_H1w', 'scale': True,
                     'relative': 'av_norm_H1w', 'slist': []},
            'u_H1test': {'list': [], 'name': 'test corrected H1 velocity error', 'abrev': 'TestCE_H1', 'scale': True},
            'u2H1test': {'list': [], 'name': 'test tentative H1 velocity error', 'abrev': 'TestTE_H1', 'scale': True},
            'd': {'list': [], 'name': 'corrected velocity L2 divergence', 'abrev': 'DC', 'scale': True, 'slist': []},
            'd2': {'list': [], 'name': 'tentative velocity L2 divergence', 'abrev': 'DT', 'scale': True, 'slist': []},
            'apg': {'list': [], 'name': 'analytic pressure gradient', 'abrev': 'APG', 'scale': True,
                    'norm': self.pg_normalization_factor},
            'av_norm_L2': {'list': [], 'name': 'analytic velocity L2 norm', 'abrev': 'AVN_L2', 'scale': False},
            'av_norm_H1': {'list': [], 'name': 'analytic velocity H1 norm', 'abrev': 'AVN_H1', 'scale': False},
            'av_norm_H1w': {'list': [], 'name': 'analytic velocity H1 norm on wall', 'abrev': 'AVN_H1w', 'scale': False},
            'a_force_wall': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AF', 'scale': False},
            'a_force_wall_normal': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AFN', 'scale': False},
            'a_force_wall_shear': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AFS', 'scale': False},
            'force_wall': {'list': [], 'name': 'force error on wall', 'abrev': 'FE', 'scale': False,
                           'relative': 'a_force_wall', 'slist': []},
            'force_wall_normal': {'list': [], 'name': 'normal force error on wall', 'abrev': 'FNE', 'scale': False,
                                  'relative': 'a_force_wall', 'slist': []},
            'force_wall_shear': {'list': [], 'name': 'shear force error on wall', 'abrev': 'FSE', 'scale': False,
                                 'relative': 'a_force_wall', 'slist': []},
            'ap_norm': {'list': [], 'name': 'analytic pressure norm', 'abrev': 'APN', 'scale': False},
            'p': {'list': [], 'name': 'pressure L2(0) error', 'abrev': 'PE', 'scale': True, 'slist': [],
                  'norm': self.p_normalization_factor},
            'pg': {'list': [], 'name': 'computed pressure gradient', 'abrev': 'PG', 'scale': True,
                   'norm': self.pg_normalization_factor},
            'pgE': {'list': [], 'name': 'computed pressure gradient error', 'abrev': 'PGE', 'scale': True,
                    'norm': self.pg_normalization_factor, 'slist': []},
            'pgEA': {'list': [], 'name': 'computed absolute pressure gradient error', 'abrev': 'PGEA',
                     'scale': True, 'norm': self.pg_normalization_factor},
            'p2': {'list': [], 'name': 'pressure tent L2(0) error', 'abrev': 'PTE', 'scale': True,
                   'slist': [], 'norm': self.p_normalization_factor},
            'pg2': {'list': [], 'name': 'computed pressure tent gradient', 'abrev': 'PTG', 'scale': True,
                    'norm': self.pg_normalization_factor},
            'pgE2': {'list': [], 'name': 'computed tent pressure tent gradient error', 'abrev': 'PTGE',
                     'scale': True, 'norm': self.pg_normalization_factor, 'slist': []},
            'pgEA2': {'list': [], 'name': 'computed absolute pressure tent gradient error',
                      'abrev': 'PTGEA', 'scale': True, 'norm': self.pg_normalization_factor}
            }

        # output files
        self.uFile = None
        self.uDiffFile = None
        self.u2File = None
        self.u2DiffFile = None
        self.dFile = None
        self.d2File = None
        self.pFile = None
        self.p2File = None
        self.pDiffFile = None
        self.p2DiffFile = None
        self.pgFile = None
        self.pg2File = None
        self.pgDiffFile = None
        self.pg2DiffFile = None
        # Dictionaries with [file, filename]
        self.fileDict = {'u': [self.uFile, 'velocity'],
                         'p': [self.pFile, 'pressure'],
                         'pg': [self.pgFile, 'pressure_grad'],
                         'd': [self.dFile, 'divergence']}
        self.fileDictTent = {'u2': [self.uFile, 'velocity_tent'],
                             'd2': [self.dFile, 'divergence_tent']}
        self.fileDictDiff = {'uD': [self.uDiffFile, 'velocity_diff'],
                             'pD': [self.pDiffFile, 'pressure_diff'],
                             'pgD': [self.pgDiffFile, 'pressure_grad_diff']}
        self.fileDictTentDiff = {'u2D': [self.u2DiffFile, 'velocity_tent_diff']}
        self.fileDictTentP = {'p2': [self.p2File, 'pressure_tent'],
                              'pg2':[self.pg2File, 'pressure_grad_tent']}
        self.fileDictTentPDiff = {'p2D': [self.p2DiffFile, 'pressure_tent_diff'],
                                  'pg2D':[self.pg2DiffFile, 'pressure_grad_tent_diff']}
        self.set_error_control_mode()

    def initialize(self, velocity_space, pressure_space, mesh, dir_name, factor, partial_solution_space, solution_space,
                   mesh_name, dt):
        print('Initializing output')
        self.str_dir_name = dir_name
        self.factor = float(factor)
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name):
            os.mkdir(self.str_dir_name)
        self.pSpace = pressure_space
        if self.doSave:
            self.vel = Function(velocity_space)
            self.D = FunctionSpace(mesh, "Lagrange", 1)
            self.divFunction = Function(self.D)
            self.pgSpace = VectorFunctionSpace(mesh, "DG", 0)
            self.pgFunction = Function(self.pgSpace)
            self.pFunction = Function(self.pSpace)
            self.initialize_xdmf_files()
        self.pg_normalization_factor.append(womersleyBC.average_analytic_pressure_grad(self.factor))
        self.p_normalization_factor.append(norm(
            interpolate(womersleyBC.average_analytic_pressure_expr(self.factor), self.pSpace), norm_type='L2'))
        self.vel_normalization_factor.append(norm(
            interpolate(womersleyBC.average_analytic_velocity_expr(self.factor), velocity_space), norm_type='L2'))
        print('Normalisation factors (vel, p, pg):', self.vel_normalization_factor[0], self.p_normalization_factor[0],
              self.pg_normalization_factor[0])
        print('Initializing error control')
        self.solutionSpace = solution_space
        self.stepsInSecond = int(round(1.0 / float(dt)))
        print("results: stepsInSecond = ", self.stepsInSecond)
        if self.doErrControl:
            if not self.isSteadyFlow:
                self.load_precomputed_bessel_functions(mesh_name, partial_solution_space)
            else:
                temp = toc()
                self.solution = interpolate(
                    Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=self.factor), solution_space)
                print("Prepared analytic solution. Time: %f" % (toc() - temp))

    def set_error_control_mode(self):
        if self.problem.d()['type'] == "steady":
            self.isSteadyFlow = True
        else:
            self.isSteadyFlow = False
        if self.problem.ECoption == "noEC":
            self.doErrControl = False
            print("Error control omitted")
        else:
            self.doErrControl = True
            if self.problem.ECoption == "test":
                self.testErrControl = True
                print("Error control in testing mode")
            else:
                print("Error control on")

# Output control========================================================================================================
    def initialize_xdmf_files(self):
        print('  Initializing output files.')
        if self.doSaveDiff:
            self.fileDict.update(self.fileDictDiff)
        if self.problem.d()['hasTentativeVel']:
            self.fileDict.update(self.fileDictTent)
            if self.doSaveDiff:
                self.fileDict.update(self.fileDictTentDiff)
        if self.problem.d()['rotation scheme']:
            self.fileDict.update(self.fileDictTentP)
            if self.doSaveDiff:
                self.fileDict.update(self.fileDictTentPDiff)
        for key, value in self.fileDict.iteritems():
            value[0] = XDMFFile(mpi_comm_world(), self.str_dir_name + "/" + value[1] + ".xdmf")
            value[0].parameters['rewrite_function_mesh'] = False  # saves lots of space (for use with static mesh)

    def update_time(self, actual_time):
        self.actual_time = round(actual_time, 3)
        self.time_list.append(self.actual_time)  # round time step to 0.001
        if self.actual_time > 0.5 and int(round(self.actual_time * 1000)) % 1000 == 0:
            self.isWholeSecond = True
            seconds = int(round(self.actual_time))
            self.second_list.append(seconds)
            self.N1 = seconds*self.stepsInSecond
            self.N0 = (seconds-1)*self.stepsInSecond
        else:
            self.isWholeSecond = False
        if not self.isSteadyFlow:
            self.solution = self.assemble_solution(self.actual_time)
            self.tc.start('analyticVnorms')
            self.analytic_v_norm_L2 = norm(self.solution, norm_type='L2')
            self.analytic_v_norm_H1 = norm(self.solution, norm_type='H1')
            self.analytic_v_norm_H1w = sqrt(assemble((inner(grad(self.solution), grad(self.solution)) +
                                                      inner(self.solution, self.solution)) * self.dsWall))
            self.listDict['av_norm_L2']['list'].append(self.analytic_v_norm_L2)
            self.listDict['av_norm_H1']['list'].append(self.analytic_v_norm_H1)
            self.listDict['av_norm_H1w']['list'].append(self.analytic_v_norm_H1w)
            self.tc.end('analyticVnorms')

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, is_tent, field):
        self.tc.start('div')
        self.divFunction.assign(project(div(field), self.D))
        # noinspection PyStatementEffect
        self.fileDict['d2' if is_tent else 'd'][0] << self.divFunction
        self.tc.end('div')

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field, t):
        self.tc.start('saveVel')
        self.vel.assign(field)
        self.fileDict['u2' if is_tent else 'u'][0] << self.vel
        if self.doSaveDiff and t > 0.0001:
            self.vel.assign((1.0 / self.vel_normalization_factor[0]) * (field - self.solution))
            self.fileDict['u2D' if is_tent else 'uD'][0] << self.vel
        self.tc.end('saveVel')

    # method for saving pressure
    def save_pressure(self, is_tent, pressure, computed_gradient):
        self.tc.start('analyticP')
        analytic_gradient = womersleyBC.analytic_pressure_grad(self.factor, self.actual_time)
        analytic_pressure = womersleyBC.analytic_pressure(self.factor, self.actual_time)
        self.sol_p = interpolate(analytic_pressure, self.pSpace)
        if not is_tent:
            self.last_analytic_pressure_norm = norm(self.sol_p, norm_type='L2')
            self.listDict['ap_norm']['list'].append(self.last_analytic_pressure_norm)
        self.tc.end('analyticP')
        self.tc.start('saveP')
        if self.doSave:
            self.fileDict['p2' if is_tent else 'p'][0] << pressure
            pg = project((1.0 / self.pg_normalization_factor[0]) * grad(pressure), self.pgSpace)
            self.pgFunction.assign(pg)
            self.fileDict['pg2' if is_tent else 'pg'][0] << self.pgFunction
            if self.doSaveDiff:
                sol_pg_expr = Expression(("0", "0", "pg"), pg=analytic_gradient / self.pg_normalization_factor[0])
                sol_pg = interpolate(sol_pg_expr, self.pgSpace)
                # plot(sol_p, title="sol")
                # plot(pressure, title="p")
                # plot(pressure - sol_p, interactive=True, title="diff")
                # exit()
                self.pFunction.assign(pressure-self.sol_p)
                self.fileDict['p2D' if is_tent else 'pD'][0] << self.pFunction
                self.pgFunction.assign(pg-sol_pg)
                self.fileDict['pg2D' if is_tent else 'pgD'][0] << self.pgFunction
        self.tc.end('saveP')
        self.tc.start('errorP')
        error = errornorm(self.sol_p, pressure, norm_type="l2", degree_rise=0)
        self.listDict['p2' if is_tent else 'p']['list'].append(error)
        print("Normalized pressure error norm:", error/self.p_normalization_factor[0])
        self.listDict['pg2' if is_tent else 'pg']['list'].append(computed_gradient)
        if not is_tent:
            self.listDict['apg']['list'].append(analytic_gradient)
        self.listDict['pgE2' if is_tent else 'pgE']['list'].append(computed_gradient-analytic_gradient)
        self.listDict['pgEA2' if is_tent else 'pgEA']['list'].append(abs(computed_gradient-analytic_gradient))
        if self.isWholeSecond:
            for key in (['pgE2', 'p2'] if is_tent else ['pgE', 'p']):
                self.listDict[key]['slist'].append(
                    sqrt(sum([i*i for i in self.listDict[key]['list'][self.N0:self.N1]])/self.stepsInSecond))
        self.tc.end('errorP')

    def report_fail(self, t):
        print("Runtime error:", sys.exc_info()[1])
        print("Traceback:")
        traceback.print_tb(sys.exc_info()[2])
        f = open(self.problem.d()['name'] + "_factor%4.2f_step_%dms_failed_at_%5.3f.report" %
                 (self.factor, self.problem.d()['dt_ms'], t), "w")
        f.write(traceback.format_exc())
        f.close()

    def compute_div(self, is_tent, velocity):
        self.tc.start('divNorm')
        div_list = self.listDict['d2' if is_tent else 'd']['list']
        div_list.append(norm(velocity, 'Hdiv0'))
        if self.isWholeSecond:
            self.listDict['d2' if is_tent else 'd']['slist'].append(
                sum([i*i for i in div_list[self.N0:self.N1]])/self.stepsInSecond)
        self.tc.end('divNorm')

# Error control=========================================================================================================

    def assemble_solution(self, t):  # returns Womersley sol for time t
        self.tc.start('assembleSol')
        sol = Function(self.solutionSpace)
        dofs2 = self.solutionSpace.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
        sol.assign(Constant(("0.0", "0.0", "0.0")))
        sol.vector()[dofs2] += self.factor * self.bessel_parabolic.vector().array()  # parabolic part of sol
        for idx in range(8):  # add modes of Womersley sol
            sol.vector()[dofs2] += self.factor * cos(self.coefs_exp[idx] * pi * t) * self.bessel_real[idx].vector().array()
            sol.vector()[dofs2] += self.factor * -sin(self.coefs_exp[idx] * pi * t) * self.bessel_complex[idx].vector().array()
        self.tc.end('assembleSol')
        return sol

    def save_solution(self, mesh_name, file_type, factor, t_start, t_end, dt, PS, solution_space):
        self.load_precomputed_bessel_functions(mesh_name, PS)
        self.factor = float(factor)
        out = None
        if file_type == 'xdmf':
            out = XDMFFile(mpi_comm_world(), 'solution_%s.xdmf' % mesh_name)
            out.parameters['rewrite_function_mesh'] = False
        elif file_type == 'hdf5':
            out = HDF5File(mpi_comm_world(), 'solution_%s.hdf5' % mesh_name, 'w')
        else:
            exit('Unsupported file type.')
        s = Function(solution_space)
        if file_type == 'hdf5':
            t = int(float(t_start)*1000)
            dt = int(float(dt)*1000)
            t_end = int(round(float(t_end)*1000))
            while t <= t_end:
                print("t = ", t)
                s.assign(self.assemble_solution(float(t)/1000.0))
                # plot(s, mode = "glyphs", title = 'saved_hdf5', interactive = True)
                out.write(s, 'sol'+str(t))
                print('saved to hdf5, sol'+str(t))
                t += dt
        elif file_type == 'xdmf':
            t = float(t_start)
            while t <= float(t_end) + DOLFIN_EPS:
                print("t = ", t)
                s.assign(self.assemble_solution(t))
                # plot(s, mode = "glyphs", title = 'saved_xdmf', interactive = True)
                out << s
                print('saved to xdmf')
                t += float(dt)

    # load precomputed Bessel functions
    def load_precomputed_bessel_functions(self, mesh_name, PS):
        self.coefs_exp = [-8, 8, 6, -6, -4, 4, 2, -2]
        f = HDF5File(mpi_comm_world(), 'precomputed/precomputed_'+mesh_name+'.hdf5', 'r')
        temp = toc()
        fce = Function(PS)
        f.read(fce, "parab")
        self.bessel_parabolic = Function(fce)
        for i in range(8):
            f.read(fce, "real%d" % i)
            self.bessel_real.append(Function(fce))
            f.read(fce, "imag%d" % i)
            self.bessel_complex.append(Function(fce))
            # plot(coefs_r_prec[i], title="coefs_r_prec", interactive=True) # reasonable values
            # plot(coefs_i_prec[i], title="coefs_i_prec", interactive=True) # reasonable values
        # plot(c0_prec,title="c0_prec",interactive=True) # reasonable values
        print("Loaded partial solution functions. Time: %f" % (toc() - temp))

    def compute_err(self, is_tent, velocity, t):
        if self.doErrControl:
            er_list_L2 = self.listDict['u2L2' if is_tent else 'u_L2']['list']
            er_list_H1 = self.listDict['u2H1' if is_tent else 'u_H1']['list']
            er_list_H1w = self.listDict['u2H1w' if is_tent else 'u_H1w']['list']
            if self.testErrControl:
                er_list_test_H1 = self.listDict['u2H1test' if is_tent else 'u_H1test']['list']
                er_list_test_L2 = self.listDict['u2L2test' if is_tent else 'u_L2test']['list']
            if self.isSteadyFlow:
                if self.testErrControl:
                    self.tc.start('errorVtest')
                    er_list_test_L2.append(errornorm(velocity, self.solution, norm_type='L2', degree_rise=0))
                    er_list_test_H1.append(errornorm(velocity, self.solution, norm_type='H1', degree_rise=0))
                    self.tc.end('errorVtest')
                self.tc.start('errorV')
                er_list_L2.append(assemble(inner(velocity - self.solution, velocity - self.solution) * dx))  # faster
                self.tc.end('errorV')
            else:
                if self.testErrControl:
                    self.tc.start('errorVtest')
                    er_list_test_L2.append(errornorm(velocity, self.solution, norm_type='L2', degree_rise=0))
                    er_list_test_H1.append(errornorm(velocity, self.solution, norm_type='H1', degree_rise=0))
                    self.tc.end('errorVtest')
                self.tc.start('errorV')
                errorL2_sq = assemble(inner(velocity - self.solution, velocity - self.solution) * dx)  # faster than errornorm
                errorH1seminorm_sq = assemble(inner(grad(velocity - self.solution), grad(velocity - self.solution)) * dx)  # faster than errornorm
                print('  H1 seminorm error:', sqrt(errorH1seminorm_sq))
                errorL2 = sqrt(errorL2_sq)
                errorH1 = sqrt(errorL2_sq + errorH1seminorm_sq)
                print("  Relative L2 error in velocity = ", errorL2 / self.analytic_v_norm_L2)
                print("  Relative H1 error in velocity = ", errorH1 / self.analytic_v_norm_H1)
                er_list_L2.append(errorL2)
                er_list_H1.append(errorH1)
                errorH1wall = sqrt(assemble((inner(grad(velocity - self.solution), grad(velocity - self.solution)) +
                                             inner(velocity - self.solution, velocity - self.solution)) * self.dsWall))
                er_list_H1w.append(errorH1wall)
                print('  Relative H1wall error:', errorH1wall / self.analytic_v_norm_H1w)
                self.tc.end('errorV')
            if self.isWholeSecond:
                self.listDict['u2L2' if is_tent else 'u_L2']['slist'].append(
                    math.sqrt(sum([i*i for i in er_list_L2[self.N0:self.N1]])/self.stepsInSecond))
                self.listDict['u2H1' if is_tent else 'u_H1']['slist'].append(
                    math.sqrt(sum([i*i for i in er_list_H1[self.N0:self.N1]])/self.stepsInSecond))
                self.listDict['u2H1w' if is_tent else 'u_H1w']['slist'].append(
                    math.sqrt(sum([i*i for i in er_list_H1w[self.N0:self.N1]])/self.stepsInSecond))

    def compute_force(self, velocity, pressure, normal, t):
        self.tc.start('errorForce')
        I = Identity(3)  # Identity tensor
        def T(p, v):
            return -p * I + 2.0 * self.nu * sym(grad(v))
        error_force = math.sqrt(
                assemble(inner((T(pressure, velocity) - T(self.sol_p, self.solution)) * normal,
                               (T(pressure, velocity) - T(self.sol_p, self.solution)) * normal) * self.dsWall))
        an_force = math.sqrt(assemble(inner(T(self.sol_p, self.solution) * normal,
                                            T(self.sol_p, self.solution) * normal) * self.dsWall))
        an_f_normal = math.sqrt(assemble(inner(inner(T(self.sol_p, self.solution) * normal, normal),
                                               inner(T(self.sol_p, self.solution) * normal, normal)) * self.dsWall))
        error_f_normal = math.sqrt(
                assemble(inner(inner((T(self.sol_p, self.solution) - T(pressure, velocity)) * normal, normal),
                               inner((T(self.sol_p, self.solution) - T(pressure, velocity)) * normal, normal)) * self.dsWall))
        an_f_shear = math.sqrt(
                assemble(inner((I - outer(normal, normal)) * T(self.sol_p, self.solution) * normal,
                               (I - outer(normal, normal)) * T(self.sol_p, self.solution) * normal) * self.dsWall))
        error_f_shear = math.sqrt(
                assemble(inner((I - outer(normal, normal)) *
                               (T(self.sol_p, self.solution) - T(pressure, velocity)) * normal,
                               (I - outer(normal, normal)) *
                               (T(self.sol_p, self.solution) - T(pressure, velocity)) * normal) * self.dsWall))
        self.listDict['a_force_wall']['list'].append(an_force)
        self.listDict['a_force_wall_normal']['list'].append(an_f_normal)
        self.listDict['a_force_wall_shear']['list'].append(an_f_shear)
        self.listDict['force_wall']['list'].append(error_force)
        self.listDict['force_wall_normal']['list'].append(error_f_normal)
        self.listDict['force_wall_shear']['list'].append(error_f_shear)
        if self.isWholeSecond:
            self.listDict['force_wall']['slist'].append(
                math.sqrt(sum([i*i for i in self.listDict['force_wall']['list'][self.N0:self.N1]])/self.stepsInSecond))
        print('  Relative force error:', error_force/an_force)
        self.tc.end('errorForce')

# Reports ==============================================================================================================
    def report(self, ):
        total = toc()
        pd = self.problem.d()

        # compare errors measured by assemble and errornorm
        if self.testErrControl:
            for e in [[self.listDict['u_L2']['list'], self.listDict['u_L2test']['list'], 'L2'],
                      [self.listDict['u_H1']['list'], self.listDict['u_H1test']['list'], 'H1']]:
                print('test ', e[2], sum([abs(e[0][i]-e[1][i]) for i in range(len(self.time_list))]))

        # report error norm, norm of div, and pressure gradients for individual time steps
        with open(self.str_dir_name + "/report_time_lines.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', escapechar='\\', quoting=csv.QUOTE_NONE)
            # report_writer.writerow(self.problem.get_metadata_to_save())
            report_writer.writerow(["name", "what", "time"] + self.time_list)
            for key in self.listDict:
                l = self.listDict[key]
                if l['list']:
                    abrev = l['abrev']
                    report_writer.writerow([pd['name'], l['name'], abrev] + l['list'])
                    if l['scale']:
                        temp_list = [i/self.factor for i in l['list']]
                        report_writer.writerow([pd['name'], "scaled " + l['name'], abrev+"s"] + temp_list)
                    if 'norm' in l:
                        # print('  norm:'+l['norm'])
                        temp_list = [i/l['norm'][0] for i in l['list']]
                        report_writer.writerow([pd['name'], "normalized " + l['name'], abrev+"n"] + temp_list)
                    if 'relative' in l:
                        norm_list = self.listDict[l['relative']]['list']
                        # print(key, len(l['list']),len(norm_list))
                        temp_list = [l['list'][i]/norm_list[i] for i in range(0, len(l['list']))]
                        self.listDict[key]['relative_list'] = temp_list
                        report_writer.writerow([pd['name'], "relative " + l['name'], abrev+"r"] + temp_list)

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
                    report_writer.writerow([pd['name'], l['name'], abrev] + value)
                    if l['scale']:
                        temp_list = [i/self.factor for i in value]
                        report_writer.writerow([pd['name'], "scaled " + l['name'], abrev+"s"] + temp_list)
                    if 'norm' in l:
                        temp_list = [i/l['norm'][0] for i in value]
                        l['normalized_list_sec'] = temp_list
                        report_writer.writerow([pd['name'], "normalized " + l['name'], abrev+"n"] + temp_list)
                    if 'relative_list' in l:
                        temp_list = []
                        # print('relative second list of', l['abrev'])
                        for sec in self.second_list:
                            N0 = (sec-1)*self.stepsInSecond
                            N1 = sec*self.stepsInSecond
                            # print(sec,  N0, N1)
                            temp_list.append(sqrt(sum([i*i for i in l['relative_list'][N0:N1]])/float(self.stepsInSecond)))
                        l['relative_list_sec'] = temp_list
                        report_writer.writerow([pd['name'], "relative " + l['name'], abrev+"r"] + temp_list)

        header_row = ["name", 'metadata', "totalTimeHours", "timeToSolve", "timeToComputeErr"]
        data_row = [pd['name'], self.problem.get_metadata_to_save(), total / 3600.0, total - self.time_erc,
                    self.time_erc]
        for key in ['u_L2', 'u_H1', 'u_H1w', 'p', 'u2L2', 'u2H1', 'u2H1w', 'p2', 'pgE', 'pgE2', 'd', 'd2', 'force_wall']:
            l = self.listDict[key]
            header_row += ['last_cycle_'+l['abrev']]
            data_row += [l['slist'][-1]] if l['slist'] else [0]
            if 'relative_list_sec' in l:
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
            self.tc.report(reportFile, pd['name'])

        # create file showing all was done well
        f = open(pd['name'] + "_factor%4.2f_step_%dms_OK.report" % (self.factor, pd['dt_ms']), "w")
        f.close()
