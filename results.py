from __future__ import print_function

__author__ = 'jh'

from dolfin import *
import os, traceback, math, csv, sys


class ResultsManager:
    def __init__(self):

        self.c0_prec = None
        self.coefs_r_prec = []
        self.coefs_i_prec = []

        self.str_dir_name = None
        self.doSave = None
        self.doSaveDiff = False
        self.doErrControl = None
        self.testErrControl = False
        self.hasTentativeVel = False
        self.isSteadyFlow = None

        self.velocity_norm = None
        self.solutionSpace = None
        self.factor = None
        self.solution = None
        self.coefs_exp = None
        self.time_erc = 0  # total time spent on measuring error
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.err_u = []
        self.err_u2 = []
        self.err_ut = []
        self.err_u2t = []
        self.isWholeSecond = False
        self.stepsInSecond = None
        self.second_list = []
        self.second_err_u = []
        self.second_err_div = []
        self.second_err_u2 = []
        self.second_err_div2 = []
        self.second_err_pg = []

        self.div_u = []
        self.div_u2 = []

        self.p_diff = []
        self.p_diff_analytic = []
        self.p_diff_err = []
        self.p_diff_err_abs = []
        self.pressure_gradient_norm = None

        self.uFile = None
        self.uDiffFile = None
        self.u2File = None
        self.u2DiffFile = None
        self.dFile = None
        self.d2File = None
        self.pFile = None
        self.pgFile = None
        self.pgDiffFile = None
        self.vel = None
        self.D = None
        self.divFunction = None
        self.PGSpace = None
        self.pgFunction = None

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
        self.uFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/velocity.xdmf")
        # saves lots of space (for use with static mesh)
        self.uFile.parameters['rewrite_function_mesh'] = False
        if self.doSaveDiff:
            self.uDiffFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/vel_sol_diff.xdmf")
            self.uDiffFile.parameters['rewrite_function_mesh'] = False
            self.pgDiffFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/pressure_grad_diff.xdmf")
            self.pgDiffFile.parameters['rewrite_function_mesh'] = False
        self.dFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/divergence.xdmf")  # maybe just compute norm
        self.dFile.parameters['rewrite_function_mesh'] = False
        self.pFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/pressure.xdmf")
        self.pFile.parameters['rewrite_function_mesh'] = False
        self.pgFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/pressure_grad.xdmf")
        self.pgFile.parameters['rewrite_function_mesh'] = False
        if self.hasTentativeVel:
            self.u2File = XDMFFile(mpi_comm_world(), self.str_dir_name + "/velocity_tent.xdmf")
            self.u2File.parameters['rewrite_function_mesh'] = False
            if self.doSaveDiff:
                self.u2DiffFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/vel_sol_diff_tent.xdmf")
                self.u2DiffFile.parameters['rewrite_function_mesh'] = False
            self.d2File = XDMFFile(mpi_comm_world(), self.str_dir_name + "/div_tent.xdmf")  # maybe just compute norm
            self.d2File.parameters['rewrite_function_mesh'] = False

    def initialize_output(self, velocity_space, mesh, dir_name, pressure_gradient_norm, velocity_norm):
        print('Initializing output')
        self.str_dir_name = dir_name
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name):
            os.mkdir(self.str_dir_name)
        if self.doSave:
            self.vel = Function(velocity_space)
            self.D = FunctionSpace(mesh, "Lagrange", 1)
            self.divFunction = Function(self.D)
            self.PGSpace = VectorFunctionSpace(mesh, "DG", 0)
            self.pgFunction = Function(self.PGSpace)
            self.initialize_xdmf_files()
        self.pressure_gradient_norm = pressure_gradient_norm
        self.velocity_norm = velocity_norm

    def update_time(self, actual_time):
        actual_time = round(actual_time, 3)
        self.time_list.append(actual_time)  # round time step to 0.001
        if actual_time > 0.5 and int(round(actual_time * 1000)) % 1000 == 0:
            self.isWholeSecond = True
            self.second_list.append(actual_time)
        else:
            self.isWholeSecond = False

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, is_tent, field):
        div_file = self.d2File if is_tent else self.dFile
        tmp = toc()
        self.divFunction.assign(project(div(field), self.D))
        div_file << self.divFunction
        print("Computed and saved divergence. Time: %f" % (toc() - tmp))

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field, t):
        vel_file = self.u2File if is_tent else self.uFile
        tmp = toc()
        self.vel.assign(field)
        vel_file << self.vel
        if self.doSaveDiff:
            vel_file = self.u2DiffFile if is_tent else self.uDiffFile
            sol = self.assemble_solution(t)
            self.vel.assign((1.0 / self.velocity_norm) * (field - sol))
            vel_file << self.vel
        print("Saved solution. Time: %f" % (toc() - tmp))

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_pressure(self, pressure, computed_gradient, analytic_gradient):
        if self.doSave:
            self.pFile << pressure
            pg = project((1.0 / self.pressure_gradient_norm) * grad(pressure), self.PGSpace)
            self.pgFunction.assign(pg)
            self.pgFile << self.pgFunction
            if self.doSaveDiff:
                sol_expr = Expression(("0", "0", "pg"), pg=analytic_gradient / self.pressure_gradient_norm)
                sol = interpolate(sol_expr, self.PGSpace)
                # plot(sol)
                # plot(pg)
                # plot(pg - sol, interactive=True)
                # exit()
                self.pgFunction.assign(pg-sol)
                self.pgDiffFile << self.pgFunction
        self.p_diff.append(computed_gradient)
        self.p_diff_analytic.append(analytic_gradient)
        self.p_diff_err.append(computed_gradient-analytic_gradient)
        self.p_diff_err_abs.append(abs(computed_gradient-analytic_gradient))
        if self.isWholeSecond:
            N1 = len(self.p_diff_err)
            N0 = N1 - self.stepsInSecond
            self.second_err_pg.append(sum(self.p_diff_err_abs[N0:N1]) / self.stepsInSecond)

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
            sec_div_list = self.second_err_div2 if isTent else self.second_err_div
            N1 = len(div_list)
            N0 = N1 - self.stepsInSecond
            sec_div_list.append(sum(div_list[N0:N1]) / self.stepsInSecond)

        print("Computed norm of divergence. Time: %f" % (toc() - tmp))

# Error control=========================================================================================================
    def initialize_error_control(self, factor, PS, V, mesh_name, dt):
        self.solutionSpace = V
        self.factor = float(factor)
        self.stepsInSecond = int(round(1.0 / float(dt)))
        print("results: stepsInSecond = ", self.stepsInSecond)
        if self.doErrControl:
            if not self.isSteadyFlow:
                self.load_precomputed_bessel_functions(mesh_name, PS)
            else:
                temp = toc()
                self.solution = interpolate(
                    Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=self.factor), V)
                print("Prepared analytic solution. Time: %f" % (toc() - temp))

    def assemble_solution(self, t):  # returns Womersley sol for time t
        tmp = toc()
        sol = Function(self.solutionSpace)
        dofs2 = self.solutionSpace.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
        sol.assign(Constant(("0.0", "0.0", "0.0")))
        sol.vector()[dofs2] += self.factor * self.c0_prec.vector().array()  # parabolic part of sol
        for idx in range(8):  # add modes of Womersley sol
            sol.vector()[dofs2] += self.factor * cos(self.coefs_exp[idx] * pi * t) * self.coefs_r_prec[idx].vector().array()
            sol.vector()[dofs2] += self.factor * -sin(self.coefs_exp[idx] * pi * t) * self.coefs_i_prec[idx].vector().array()
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
        self.c0_prec = Function(fce)
        for i in range(8):
            f.read(fce, "real%d" % i)
            self.coefs_r_prec.append(Function(fce))
            f.read(fce, "imag%d" % i)
            self.coefs_i_prec.append(Function(fce))
            # plot(coefs_r_prec[i], title="coefs_r_prec", interactive=True) # reasonable values
            # plot(coefs_i_prec[i], title="coefs_i_prec", interactive=True) # reasonable values
        # plot(c0_prec,title="c0_prec",interactive=True) # reasonable values
        print("Loaded partial solution functions. Time: %f" % (toc() - temp))

    def compute_err(self, is_tent, velocity, t):
        if self.doErrControl:
            er_list = self.err_u2 if is_tent else self.err_u
            if self.testErrControl:
                er_list_test = self.err_u2t if is_tent else self.err_ut
            tmp = toc()
            if self.isSteadyFlow:
                if self.testErrControl:
                    er_list_test.append(errornorm(velocity, self.solution, norm_type='l2', degree_rise=0))
                er_list.append(assemble(inner(velocity - self.solution, velocity - self.solution) * dx))  # faster
            else:
                sol = self.assemble_solution(t)
                if self.testErrControl:
                    er_list_test.append(errornorm(velocity, sol, norm_type='l2', degree_rise=0))
                error = assemble(inner(velocity - sol, velocity - sol) * dx)
                print("  Error in velocity = ", math.sqrt(error))
                er_list.append(error)  # faster
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

            total_err_pg = sum(self.p_diff_err)
            avg_err_pg = total_err_pg * dt / ttime
            N1 = len(self.p_diff_err)
            N0 = int(round(N1 - 1.0/dt))
            last_cycle_err_pg = sum(self.p_diff_err[N0:N1])*dt

            avg_err_u = total_err_u / math.sqrt(len(self.time_list))
            avg_err_u2 = total_err_u2 / math.sqrt(len(self.time_list))

            last_cycle = self.time_list[N0:N1]
            # print("N0,N1: ",N0, N1 ," len: ",len(last_cycle), " list: ",last_cycle)
            last_cycle_err_u = math.sqrt(sum(self.err_u[N0:N1]) * dt)
            last_cycle_div = sum(self.div_u[N0:N1]) * dt
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
            if self.doErrControl:
                report_writer.writerow([str_name] + ["corrected_error"] + [str_name + "_CE"] + self.err_u)
                self.err_u = [i/self.factor for i in self.err_u]
                report_writer.writerow([str_name] + ["corrected_error_scaled"] + [str_name + "_CEs"] + self.err_u)
                if self.testErrControl:
                    report_writer.writerow([str_name] + ["test_corrected_errornorm"] + [str_name + "_TCE"] + self.err_ut)
                if self.hasTentativeVel:
                    report_writer.writerow([str_name] + ["tentative_error"] + [str_name + "_TE"] + self.err_u2)
                    self.err_u2 = [i/self.factor for i in self.err_u2]
                    report_writer.writerow([str_name] + ["tentative_error_scaled"] + [str_name + "_TEs"] + self.err_u2)
                    if self.testErrControl:
                        report_writer.writerow([str_name] + ["test_tentative_errornorm"] + [str_name + "_TTE"] + self.err_u2t)

            report_writer.writerow([str_name] + ["divergence_corrected"] + [str_name + "_DC"] + self.div_u)
            if self.hasTentativeVel:
                report_writer.writerow([str_name] + ["divergence_tentative"] + [str_name + "_DT"] + self.div_u2)
            report_writer.writerow([str_name] + ["analytic_pressure_gradient"] + ["analytic"] + self.p_diff_analytic)
            report_writer.writerow([str_name] + ["computed_pressure_gradient"] + [str_name + "_PG"] + self.p_diff)
            report_writer.writerow([str_name] + ["pressure_gradient_error"] + [str_name + "_PGE"] + self.p_diff_err)
            report_writer.writerow([str_name] + ["pressure_gradient_error_abs"] + [str_name + "_PGEA"] + self.p_diff_err_abs)
            p_diff_analytic = [i/self.factor for i in self.p_diff_analytic]
            p_diff = [i/self.factor for i in self.p_diff]
            p_diff_err = [i/self.factor for i in self.p_diff_err]
            p_diff_err_abs = [i/self.factor for i in self.p_diff_err_abs]
            report_writer.writerow([str_name] + ["scaled_analytic_pressure_gradient"] + ["analytic_scaled"] + p_diff_analytic)
            report_writer.writerow([str_name] + ["scaled_computed_pressure_gradient"] + [str_name + "_PGs"] + p_diff)
            report_writer.writerow([str_name] + ["scaled_pressure_gradient_error"] + [str_name + "_PGEs"] + p_diff_err)
            report_writer.writerow([str_name] + ["scaled_pressure_gradient_error_abs"] + [str_name + "_PGEAs"] + p_diff_err_abs)
            self.p_diff_analytic = [i/self.pressure_gradient_norm for i in self.p_diff_analytic]
            self.p_diff = [i/self.pressure_gradient_norm for i in self.p_diff]
            self.p_diff_err = [i/self.pressure_gradient_norm for i in self.p_diff_err]
            self.p_diff_err_abs = [i/self.pressure_gradient_norm for i in self.p_diff_err_abs]
            report_writer.writerow([str_name] + ["normalized_analytic_pressure_gradient"] + ["analytic_normalized"] + self.p_diff_analytic)
            report_writer.writerow([str_name] + ["normalized_computed_pressure_gradient"] + [str_name + "_PGn"] + self.p_diff)
            report_writer.writerow([str_name] + ["normalized_pressure_gradient_error"] + [str_name + "_PGEn"] + self.p_diff_err)
            report_writer.writerow([str_name] + ["normalized_pressure_gradient_error_abs"] + [str_name + "_PGEAn"] + self.p_diff_err_abs)

        # report error norm, norm of div, and pressure gradients averaged over seconds
        with open(self.str_dir_name + "/report_seconds.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(["name"] + ["time"] + self.second_list)
            if self.doErrControl:
                report_writer.writerow([str_name] + ["corrected_error"] + self.second_err_u)
                self.second_err_u = [i/self.factor for i in self.second_err_u]
                report_writer.writerow([str_name] + ["corrected_error_scaled"] + self.second_err_u)
                if self.hasTentativeVel:
                    report_writer.writerow([str_name] + ["tentative_error"] + self.second_err_u2)
                    self.second_err_u2 = [i/self.factor for i in self.second_err_u2]
                    report_writer.writerow([str_name] + ["tentative_error_scaled"] + self.second_err_u2)

            report_writer.writerow([str_name] + ["divergence_corrected"] + self.second_err_div)
            if self.hasTentativeVel:
                report_writer.writerow([str_name] + ["divergence_tentative"] + self.second_err_div2)
            report_writer.writerow([str_name] + ["pressure_gradient_error"] + self.second_err_pg)
            self.second_err_pg = [i/self.factor for i in self.second_err_pg]
            report_writer.writerow([str_name] + ["pressure_gradient_error_scaled"] + self.second_err_pg)

        # report without header
        with open(self.str_dir_name + "/report.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [mesh] + [str_solver] +
                [factor] + [ttime] + [dt] + [total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] +
                [avg_err_u] + [avg_err_u2] + [last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] +
                [last_cycle_div2] + [last_cycle_err_min] + [last_cycle_err_max] + [last_cycle_err_min2] +
                [last_cycle_err_max2] + [avg_err_pg] + [last_cycle_err_pg])

        # report with header
        with open(self.str_dir_name + "/report_h.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["problem"] + ["name"] + ["type"] + ["method"] + ["mesh_name"] + ["mesh"] + ["solver"] + ["factor"] +
                ["time"] + ["dt"] + ["timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"] +
                ["avg_err_u"] + ["avg_err_u2"] + ["last_cycle_err_u"] + ["last_cycle_err_u2"] + ["last_cycle_div"] +
                ["last_cycle_div2"] + ["last_cycle_err_min"] + ["last_cycle_err_max"] + ["last_cycle_err_min2"] +
                ["last_cycle_err_max2"] + ["avg_err_pg"] + ["last_cycle_err_pg"])
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [mesh] + [str_solver] +
                [factor] + [ttime] + [dt] + [total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] +
                [avg_err_u] + [avg_err_u2] + [last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] +
                [last_cycle_div2] + [last_cycle_err_min] + [last_cycle_err_max] + [last_cycle_err_min2] +
                [last_cycle_err_max2] + [avg_err_pg] + [last_cycle_err_pg])

        # create file showing all was done well
        f = open(str_name + "_factor%4.2f_step_%dms_OK.report" % (factor, dt * 1000), "w")
        f.close()




