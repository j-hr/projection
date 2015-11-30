from __future__ import print_function

__author__ = 'jh'

from dolfin import *
import os, traceback, math, csv, sys
import womersleyBC


class ResultsManager:
    def __init__(self):

        # partial Bessel functions and coefficients
        self.bessel_parabolic = None
        self.bessel_real = []
        self.bessel_complex = []
        self.coefs_exp = None

        self.str_dir_name = None
        self.doSave = None
        self.doSaveDiff = False
        self.doErrControl = None
        self.testErrControl = False
        self.hasTentativeVel = False
        self.hasTentativePressure = False
        self.isSteadyFlow = None

        self.velocity_norm = None
        self.pressure_gradient_norm = []
        self.solutionSpace = None
        self.factor = None
        self.solution = None
        self.time_erc = 0  # total time spent on measuring error
        self.actual_time = None
        self.isWholeSecond = False
        self.stepsInSecond = None
        self.vel = None
        self.D = None
        self.divFunction = None
        self.pSpace = None
        self.pFunction = None
        self.pgSpace = None
        self.pgFunction = None

        # lists
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.av_norm = []
        self.ap_norm = []
        self.err_u = []
        self.err_u2 = []
        self.err_ut = []
        self.err_u2t = []
        self.second_list = []
        self.second_err_u = []
        self.second_div = []
        self.second_err_u2 = []
        self.second_div2 = []
        self.second_err_p = []
        self.second_err_p2 = []
        self.second_err_pg = []
        self.second_err_pg2 = []
        self.div_u = []
        self.div_u2 = []
        self.p_err = []
        self.p_err2 = []
        self.pg = []
        self.pg2 = []
        self.pg_analytic = []
        self.pg_err = []
        self.pg_err2 = []
        self.pg_err_abs = []
        self.pg_err_abs2 = []
        # dictionary of data lists {list, name, abbreviation, add scaled row to report}
        # normalisation coefficients (time-independent) are added to some lists to be used in normalized data series
        # norm lists (time-dependent normalisation coefficients) are added to some lists to be used in relative data
        #  series (to remove natural pulsation of error due to change in volume flow rate)
        # L2(0) means L2 difference of pressures taken with zero average
        self.listDict = {
            'u': {'list': self.err_u, 'name': 'corrected velocity L2 error', 'abrev': 'CE', 'scale': True,
                  'relative': 'av_norm'},
            'u2': {'list': self.err_u2, 'name': 'tentative velocity L2 error', 'abrev': 'TE', 'scale': True,
                   'relative': 'av_norm'},
            'u_test': {'list': self.err_ut, 'name': 'test corrected L2 velocity error', 'abrev': 'TCE', 'scale': True},
            'u2test': {'list': self.err_u2t, 'name': 'test tentative L2 velocity error', 'abrev': 'TTE', 'scale': True},
            'd': {'list': self.div_u, 'name': 'corrected velocity L2 divergence', 'abrev': 'DC', 'scale': True},
            'd2': {'list': self.div_u2, 'name': 'tentative velocity L2 divergence', 'abrev': 'DT', 'scale': True},
            'apg': {'list': self.pg_analytic, 'name': 'analytic pressure gradient', 'abrev': 'APG', 'scale': True,
                    'norm': self.pressure_gradient_norm},
            'av_norm': {'list': self.av_norm, 'name': 'analytic velocity norm', 'abrev': 'APN', 'scale': False},
            'ap_norm': {'list': self.ap_norm, 'name': 'analytic pressure norm', 'abrev': 'APN', 'scale': False},
            'p': {'list': self.p_err, 'name': 'pressure L2(0) error', 'abrev': 'PE', 'scale': True,
                  'relative': 'ap_norm'},
            'pg': {'list': self.pg, 'name': 'computed pressure gradient', 'abrev': 'PG', 'scale': True,
                   'norm': self.pressure_gradient_norm},
            'pgE': {'list': self.pg_err, 'name': 'computed pressure gradient error', 'abrev': 'PGE', 'scale': True,
                    'norm': self.pressure_gradient_norm},
            'pgEA': {'list': self.pg_err_abs, 'name': 'computed absolute pressure gradient error', 'abrev': 'PGEA',
                     'scale': True, 'norm': self.pressure_gradient_norm},
            'p2': {'list': self.p_err2, 'name': 'pressure tent L2(0) error', 'abrev': 'PTE', 'scale': True,
                   'relative': 'ap_norm'},
            'pg2': {'list': self.pg2, 'name': 'computed pressure tent gradient', 'abrev': 'PTG', 'scale': True,
                    'norm': self.pressure_gradient_norm},
            'pgE2': {'list': self.pg_err2, 'name': 'computed tent pressure tent gradient error', 'abrev': 'PTGE',
                     'scale': True, 'norm': self.pressure_gradient_norm},
            'pgEA2': {'list': self.pg_err_abs2, 'name': 'computed absolute pressure tent gradient error',
                      'abrev': 'PTGEA', 'scale': True, 'norm': self.pressure_gradient_norm}
            }
        # dictionary of data lists for cycle-averaged values {list}
        # use same keys as in list dict for corresponding values
        self.secondListDict = {'u': self.second_err_u,
                               'u2': self.second_err_u2,
                               'd': self.second_div,
                               'd2': self.second_div2,
                               'p': self.second_err_p,
                               'p2': self.second_err_p2,
                               'pgE': self.second_err_pg,
                               'pgE2': self.second_err_pg2}

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
                         'pg':[self.pgFile, 'pressure_grad'],
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
        self.pressure_gradient_norm.append(womersleyBC.average_analytic_pressure_grad(self.factor))
        self.velocity_norm = womersleyBC.average_analytic_velocity(self.factor)
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

    def set_save_mode(self, option):
        if option == 'save' or option == 'diff':
            self.doSave = True
            if option == 'diff':
                self.doSaveDiff = True
                print('Saving velocity differences.')
            print('Saving solution ON.')
        elif option == 'noSave':
            self.doSave = False
            print('Saving solution OFF.')
        else:
            exit('Wrong parameter save_results, should be \"save\" o \"noSave\".')

    def set_error_control_mode(self, option, str_type):
        if str_type == "steady":
            self.isSteadyFlow = True
        else:
            self.isSteadyFlow = False
        if option == "noEC":
            self.doErrControl = False
            print("Error control omitted")
        else:
            self.doErrControl = True
            if option == "test":
                self.testErrControl = True
                print("Error control in testing mode")
            else:
                print("Error control on")

# Output control========================================================================================================
    def initialize_xdmf_files(self):
        print('  Initializing output files.')
        if self.doSaveDiff:
            self.fileDict.update(self.fileDictDiff)
        if self.hasTentativeVel:
            self.fileDict.update(self.fileDictTent)
            if self.doSaveDiff:
                self.fileDict.update(self.fileDictTentDiff)
        if self.hasTentativePressure:
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
            self.second_list.append(self.actual_time)
        else:
            self.isWholeSecond = False

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, is_tent, field):
        tmp = toc()
        self.divFunction.assign(project(div(field), self.D))
        self.fileDict['d2' if is_tent else 'd'][0] << self.divFunction
        print("Computed and saved divergence. Time: %f" % (toc() - tmp))

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field, t):
        tmp = toc()
        self.vel.assign(field)
        self.fileDict['u2' if is_tent else 'u'][0] << self.vel
        if self.doSaveDiff:
            sol = self.assemble_solution(t)
            self.vel.assign((1.0 / self.velocity_norm) * (field - sol))
            self.fileDict['u2D' if is_tent else 'uD'][0] << self.vel
        print("Saved solution. Time: %f" % (toc() - tmp))

    # method for saving pressure
    def save_pressure(self, is_tent, pressure, computed_gradient):
        analytic_gradient = womersleyBC.analytic_pressure_grad(self.factor, self.actual_time)
        analytic_pressure = womersleyBC.analytic_pressure(self.factor, self.actual_time)
        sol_p = interpolate(analytic_pressure, self.pSpace)
        if not is_tent:
            self.listDict['ap_norm']['list'].append(norm(sol_p, norm_type='L2'))
        if self.doSave:
            self.fileDict['p2' if is_tent else 'p'][0] << pressure
            pg = project((1.0 / self.pressure_gradient_norm[0]) * grad(pressure), self.pgSpace)
            self.pgFunction.assign(pg)
            self.fileDict['pg2' if is_tent else 'pg'][0] << self.pgFunction
            if self.doSaveDiff:
                sol_pg_expr = Expression(("0", "0", "pg"), pg=analytic_gradient / self.pressure_gradient_norm[0])
                sol_pg = interpolate(sol_pg_expr, self.pgSpace)
                # plot(sol_p, title="sol")
                # plot(pressure, title="p")
                # plot(pressure - sol_p, interactive=True, title="diff")
                # exit()
                self.pFunction.assign(pressure-sol_p)
                self.fileDict['p2D' if is_tent else 'pD'][0] << self.pFunction
                self.pgFunction.assign(pg-sol_pg)
                self.fileDict['pg2D' if is_tent else 'pgD'][0] << self.pgFunction
        p_solution_expr = analytic_pressure
        p_solution = interpolate(p_solution_expr, self.pSpace)
        self.listDict['p2' if is_tent else 'p']['list'].append(errornorm(p_solution, pressure, norm_type="l2", degree_rise=0))
        self.listDict['pg2' if is_tent else 'pg']['list'].append(computed_gradient)
        if not is_tent:
            self.listDict['apg']['list'].append(analytic_gradient)
        self.listDict['pgE2' if is_tent else 'pgE']['list'].append(computed_gradient-analytic_gradient)
        self.listDict['pgEA2' if is_tent else 'pgEA']['list'].append(abs(computed_gradient-analytic_gradient))
        if self.isWholeSecond:
            if is_tent:
                N1 = len(self.p_err2)
                N0 = N1 - self.stepsInSecond
                self.second_err_pg2.append(sum(self.pg_err_abs2[N0:N1]) / self.stepsInSecond)
                self.second_err_p2.append(sum(self.p_err2[N0:N1]) / self.stepsInSecond)
            else:
                N1 = len(self.p_err)
                N0 = N1 - self.stepsInSecond
                self.second_err_pg.append(sum(self.pg_err_abs[N0:N1]) / self.stepsInSecond)
                self.second_err_p.append(sum(self.p_err[N0:N1]) / self.stepsInSecond)

    def report_fail(self, str_name, dt, t):
        print("Runtime error:", sys.exc_info()[1])
        print("Traceback:")
        traceback.print_tb(sys.exc_info()[2])
        f = open(str_name + "_factor%4.2f_step_%dms_failed_at_%5.3f.report" % (self.factor, dt * 1000, t), "w")
        f.write(traceback.format_exc())
        f.close()

    def compute_div(self, isTent, velocity):
        div_list = self.div_u2 if isTent else self.div_u
        tmp = toc()
        div_list.append(norm(velocity, 'Hdiv0'))
        if self.isWholeSecond:
            sec_div_list = self.second_div2 if isTent else self.second_div
            N1 = len(div_list)
            N0 = N1 - self.stepsInSecond
            sec_div_list.append(sum(div_list[N0:N1]) / self.stepsInSecond)

        print("Computed norm of divergence. Time: %f" % (toc() - tmp))

# Error control=========================================================================================================

    def assemble_solution(self, t):  # returns Womersley sol for time t
        tmp = toc()
        sol = Function(self.solutionSpace)
        dofs2 = self.solutionSpace.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
        sol.assign(Constant(("0.0", "0.0", "0.0")))
        sol.vector()[dofs2] += self.factor * self.bessel_parabolic.vector().array()  # parabolic part of sol
        for idx in range(8):  # add modes of Womersley sol
            sol.vector()[dofs2] += self.factor * cos(self.coefs_exp[idx] * pi * t) * self.bessel_real[idx].vector().array()
            sol.vector()[dofs2] += self.factor * -sin(self.coefs_exp[idx] * pi * t) * self.bessel_complex[idx].vector().array()
        print("Assembled analytic solution. Time: %f" % (toc() - tmp))
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
            er_list = self.listDict['u2' if is_tent else 'u']['list']
            if self.testErrControl:
                er_list_test = self.listDict['u2test' if is_tent else 'u_test']['list']
            tmp = toc()
            if self.isSteadyFlow:
                if self.testErrControl:
                    er_list_test.append(errornorm(velocity, self.solution, norm_type='l2', degree_rise=0))
                er_list.append(assemble(inner(velocity - self.solution, velocity - self.solution) * dx))  # faster
            else:
                sol = self.assemble_solution(t)
                if not is_tent:
                    self.listDict['av_norm']['list'].append(norm(sol, norm_type='L2'))
                if self.testErrControl:
                    er_list_test.append(errornorm(velocity, sol, norm_type='l2', degree_rise=0))
                error = assemble(inner(velocity - sol, velocity - sol) * dx)  # faster than errornorm
                print("  Error in velocity = ", math.sqrt(error))
                er_list.append(error)
            if self.isWholeSecond:
                sec_err_list = self.second_err_u2 if is_tent else self.second_err_u
                N1 = len(er_list)
                N0 = N1 - self.stepsInSecond
                sec_err_list.append(math.sqrt(sum(er_list[N0:N1]) / self.stepsInSecond))
            terc = toc() - tmp
            self.time_erc += terc
            print("Computed errornorm. Time: %f, Total: %f" % (terc, self.time_erc))

# Reports ==============================================================================================================
    def report(self, dt, ttime, str_name, str_type, str_method, mesh_name, mesh, factor, str_solver):
        total = toc()
        total_err_u = 0
        total_err_u2 = 0
        avg_err_u = 0
        avg_err_u2 = 0
        last_cycle_err_u = 0
        last_cycle_err_u2 = 0
        last_cycle_div = 0
        last_cycle_div2 = 0
        last_cycle_err_min = 0
        last_cycle_err_max = 0
        last_cycle_err_min2 = 0
        last_cycle_err_max2 = 0
        total_err_pg = 0            # pressure gradient error
        avg_err_pg = 0
        last_cycle_err_pg = 0
        if self.doErrControl:
            total_err_u = math.sqrt(sum(self.err_u))
            total_err_u2 = math.sqrt(sum(self.err_u2))

            total_err_pg = sum(self.pg_err_abs)
            avg_err_pg = total_err_pg * dt / ttime
            N1 = len(self.pg_err_abs)
            N0 = N1 - self.stepsInSecond
            last_cycle_err_pg = sum(self.pg_err_abs[N0:N1])*dt

            avg_err_u = total_err_u / math.sqrt(len(self.time_list))
            avg_err_u2 = total_err_u2 / math.sqrt(len(self.time_list))

            # last_cycle = self.time_list[N0:N1]
            # print("N0,N1: ", N0, N1, " len: ",len(last_cycle), " list: ",last_cycle)
            last_cycle_err_u = math.sqrt(sum(self.err_u[N0:N1]) * dt)
            last_cycle_div = sum(self.div_u[N0:N1]) * dt
            last_cycle_err_p = sum(self.p_err[N0:N1]) * dt
            last_cycle_err_min = math.sqrt(min(self.err_u[N0:N1]))
            last_cycle_err_max = math.sqrt(max(self.err_u[N0:N1]))
            if self.hasTentativeVel:
                last_cycle_err_u2 = math.sqrt(sum(self.err_u2[N0:N1]) * dt)
                last_cycle_div2 = sum(self.div_u2[N0:N1]) * dt
                last_cycle_err_min2 = math.sqrt(min(self.err_u2[N0:N1]))
                last_cycle_err_max2 = math.sqrt(max(self.err_u2[N0:N1]))

            self.err_u = [math.sqrt(i) for i in self.err_u]
            self.err_u2 = [math.sqrt(i) for i in self.err_u2]

        # report error norm, norm of div, and pressure gradients for individual time steps
        with open(self.str_dir_name + "/report_time_lines.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(["name"] + ["what"] + ["time"] + self.time_list)
            for key in self.listDict:
                l = self.listDict[key]
                if l['list']:
                    abrev = str_name + "_" + l['abrev']
                    report_writer.writerow([str_name] + [l['name']] + [abrev] + l['list'])
                    if l['scale']:
                        temp_list = [i/self.factor for i in l['list']]
                        report_writer.writerow([str_name] + ["scaled " + l['name']] + [abrev+"s"] + temp_list)
                    if 'norm' in l:
                        # print('  norm:'+l['norm'])
                        temp_list = [i/l['norm'][0] for i in l['list']]
                        report_writer.writerow([str_name] + ["normalized " + l['name']] + [abrev+"n"] + temp_list)
                    if 'relative' in l:
                        norm_list = self.listDict[l['relative']]['list']
                        temp_list = [l['list'][i]/norm_list[i] for i in range(0, len(l['list']))]
                        report_writer.writerow([str_name] + ["relative " + l['name']] + [abrev+"r"] + temp_list)

        # report error norm, norm of div, and pressure gradients averaged over seconds
        with open(self.str_dir_name + "/report_seconds.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(["name"] + ["what"] + ["time"] + self.second_list)
            for key, value in self.secondListDict.iteritems():
                l = self.listDict[key]
                if value:
                    abrev = str_name + "_" + l['abrev']
                    report_writer.writerow([str_name, l['name'], abrev] + value)
                    if l['scale']:
                        temp_list = [i/self.factor for i in value]
                        report_writer.writerow([str_name] + ["scaled " + l['name']] + [abrev+"s"] + temp_list)
                    if 'norm' in l:
                        temp_list = [i/l['norm'][0] for i in value]
                        report_writer.writerow([str_name] + ["normalized " + l['name']] + [abrev+"n"] + temp_list)

        # report without header
        with open(self.str_dir_name + "/report.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [str_solver] +
                [factor] + [ttime] + [dt] + [total/3600.0] + [total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] +
                [avg_err_u] + [avg_err_u2] + [last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_err_p] + [last_cycle_div] +
                [last_cycle_div2] + [last_cycle_err_min] + [last_cycle_err_max] + [last_cycle_err_min2] +
                [last_cycle_err_max2] + [avg_err_pg] + [last_cycle_err_pg] + [mesh])

        # report with header
        with open(self.str_dir_name + "/report_h.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["problem"] + ["name"] + ["type"] + ["method"] + ["mesh_name"] + ["solver"] + ["factor"] +
                ["time"] + ["dt"] + ["totalTimeHours"] + ["timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"] +
                ["avg_err_u"] + ["avg_err_u2"] + ["last_cycle_err_u"] + ["last_cycle_err_u2"] + ["last_cycle_err_p"] + ["last_cycle_div"] +
                ["last_cycle_div2"] + ["last_cycle_err_min"] + ["last_cycle_err_max"] + ["last_cycle_err_min2"] +
                ["last_cycle_err_max2"] + ["avg_err_pg"] + ["last_cycle_err_pg"] + ["mesh"])
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [str_solver] +
                [factor] + [ttime] + [dt] + [total/3600.0] + [total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] +
                [avg_err_u] + [avg_err_u2] + [last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_err_p] + [last_cycle_div] +
                [last_cycle_div2] + [last_cycle_err_min] + [last_cycle_err_max] + [last_cycle_err_min2] +
                [last_cycle_err_max2] + [avg_err_pg] + [last_cycle_err_pg] + [mesh])

        # create file showing all was done well
        f = open(str_name + "_factor%4.2f_step_%dms_OK.report" % (factor, dt * 1000), "w")
        f.close()




