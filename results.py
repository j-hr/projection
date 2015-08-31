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
        self.doErrControl = None
        self.measure_time = None
        self.hasTentativeVel = False

        self.solution = None
        self.coefs_exp = None
        self.time_erc = 0  # total time spent on measuring error
        self.time_list = []  # list of times, when error is  measured (used in report)
        self.err_u = []
        self.err_u2 = []

        self.div_u = []
        self.div_u2 = []

        self.uFile = None
        self.u2File = None
        self.dFile = None
        self.d2File = None
        self.pFile = None
        self.vel = None
        self.D = None
        self.divFunction = None

    def set_save_mode(self, option):
        if option == 'save':
            self.doSave = True
            print('Saving solution ON.')
        elif option == 'noSave':
            self.doSave = False
            print('Saving solution OFF.')
        else:
            exit('Wrong parameter save_results, should be \"save\" o \"noSave\".')

    def set_error_control_mode(self, option, str_type):
        if option == "0":
            self.doErrControl = False
            print("Error control omitted")
        else:
            self.doErrControl = True
            if option == "-1":
                self.measure_time = 0.5 if str_type == "steady" else 1
            else:
                self.measure_time = float(sys.argv[7])
            print("Error control from:       %4.2f s" % self.measure_time)

# Output control========================================================================================================
    def initialize_xdmf_files(self):
        print('  Initializing output files.')
        self.uFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/velocity.xdmf")
        # saves lots of space (for use with static mesh)
        self.uFile.parameters['rewrite_function_mesh'] = False
        self.dFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/divergence.xdmf")  # maybe just compute norm
        self.dFile.parameters['rewrite_function_mesh'] = False
        self.pFile = XDMFFile(mpi_comm_world(), self.str_dir_name + "/pressure.xdmf")
        self.pFile.parameters['rewrite_function_mesh'] = False
        if self.hasTentativeVel:
            self.u2File = XDMFFile(mpi_comm_world(), self.str_dir_name + "/velocity_tent.xdmf")
            self.u2File.parameters['rewrite_function_mesh'] = False
            self.d2File = XDMFFile(mpi_comm_world(), self.str_dir_name + "/div_tent.xdmf")  # maybe just compute norm
            self.d2File.parameters['rewrite_function_mesh'] = False

    def initialize_output(self, velocity_space, mesh, dir_name):
        print('Initializing output')
        self.str_dir_name = dir_name
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name):
            os.mkdir(self.str_dir_name)
        if self.doSave:
            self.vel = Function(velocity_space)
            self.D = FunctionSpace(mesh, "Lagrange", 1)
            self.divFunction = Function(self.D)
            self.initialize_xdmf_files()

    def do_compute_error(self, time_step):
        if not self.doErrControl:
            return False
        return round(time_step, 3) >= self.measure_time

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, is_tent, field):
        div_file = self.d2File if is_tent else self.dFile
        tmp = toc()
        self.divFunction.assign(project(div(field), self.D))
        div_file << self.divFunction
        print("Computed and saved divergence. Time: %f" % (toc() - tmp))

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, is_tent, field):
        vel_file = self.u2File if is_tent else self.uFile
        tmp = toc()
        self.vel.assign(field)
        vel_file << self.vel
        print("Saved solution. Time: %f" % (toc() - tmp))

    def report_fail(self, str_name, factor, dt, t):
        print("Runtime error:", sys.exc_info()[1])
        print("Traceback:")
        traceback.print_tb(sys.exc_info()[2])
        f = open(str_name + "_factor%4.2f_step_%dms_failed_at_%5.3f.report" % (factor, dt * 1000, t), "w")
        f.write(traceback.format_exc())
        f.close()

    def compute_div(self, isTent, velocity):
        div_list = self.div_u2 if isTent else self.div_u
        tmp = toc()
        div_list.append(norm(velocity, 'Hdiv0'))
        print("Computed norm of divergence. Time: %f" % (toc() - tmp))

# Error control=========================================================================================================
    def initialize_error_control(self, str_type, factor, PS, V, mesh_name):
        if self.doErrControl:
            if str_type == "pulse0" or str_type == "pulsePrec":
                self.load_precomputed_bessel_functions(mesh_name, PS)
            if str_type == "steady":
                temp = toc()
                self.solution = interpolate(
                    Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=factor), V)
                print("Prepared analytic solution. Time: %f" % (toc() - temp))

    def assemble_solution(self, t, V, factor):  # returns Womersley sol for time t
        factor = float(factor)
        tmp = toc()
        sol = Function(V)
        dofs2 = V.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
        sol.assign(Constant(("0.0", "0.0", "0.0")))
        sol.vector()[dofs2] += factor * self.c0_prec.vector().array()  # parabolic part of sol
        for idx in range(8):  # add modes of Womersley sol
            sol.vector()[dofs2] += factor * cos(self.coefs_exp[idx] * pi * t) * self.coefs_r_prec[idx].vector().array()
            sol.vector()[dofs2] += factor * -sin(self.coefs_exp[idx] * pi * t) * self.coefs_i_prec[idx].vector().array()
        print("Assembled analytic solution. Time: %f" % (toc() - tmp))
        return sol

    def save_solution(self, mesh_name, file_type, factor, t_start, t_end, dt, PS, solution_space):
        self.load_precomputed_bessel_functions(mesh_name, PS)
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
                s.assign(self.assemble_solution(float(t)/1000.0, solution_space, factor))
                # plot(s, mode = "glyphs", title = 'saved_hdf5', interactive = True)
                out.write(s, 'sol'+str(t))
                print('saved to hdf5, sol'+str(t))
                t += dt
        elif file_type == 'xdmf':
            t = float(t_start)
            while t <= float(t_end) + DOLFIN_EPS:
                print("t = ", t)
                s.assign(self.assemble_solution(t, solution_space, factor))
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

    def compute_err(self, is_tent, velocity, t, str_type, V, factor):
        if self.doErrControl:
            er_list = self.err_u2 if is_tent else self.err_u
            if len(self.time_list) == 0 or (self.time_list[-1] < round(t, 3)):  # add only once per time step
                self.time_list.append(round(t, 3))  # round time step to 0.001
            tmp = toc()
            if str_type == "steady":
                # er_list.append(pow(errornorm(velocity, solution, norm_type='l2', degree_rise=0),2)) # slower, reliable
                er_list.append(assemble(inner(velocity - self.solution, velocity - self.solution) * dx))  # faster
            elif (str_type == "pulse0") or (str_type == "pulsePrec"):
                # er_list.append(pow(errornorm(velocity, assembleSolution(t), norm_type='l2', degree_rise=0),2))
                sol = self.assemble_solution(t, V, factor)
                er_list.append(assemble(inner(velocity - sol, velocity - sol) * dx))  # faster
            terc = toc() - tmp
            self.time_erc += terc
            print("Computed errornorm. Time: %f, Total: %f" % (terc, self.time_erc))

# Reports ==============================================================================================================
    def report(self, dt, ttime, str_name, str_type, str_method, mesh_name, mesh, factor):
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
        if self.doErrControl:
            total_err_u = math.sqrt(sum(self.err_u))
            total_err_u2 = math.sqrt(sum(self.err_u2))
            avg_err_u = total_err_u / math.sqrt(len(self.time_list))
            avg_err_u2 = total_err_u2 / math.sqrt(len(self.time_list))
            if ttime >= self.measure_time + 1 - DOLFIN_EPS:
                N = 1.0 / dt
                N0 = int(round(len(self.time_list) - N))
                N1 = int(round(len(self.time_list)))
                # last_cycle = time_list[N0:N1]
                # print("N: ",N," len: ",len(last_cycle), " list: ",last_cycle)
                last_cycle_err_u = math.sqrt(sum(self.err_u[N0:N1]) / N)
                last_cycle_div = sum(self.div_u[N0:N1]) / N
                last_cycle_err_min = math.sqrt(min(self.err_u[N0:N1]))
                last_cycle_err_max = math.sqrt(max(self.err_u[N0:N1]))
                if self.hasTentativeVel:
                    last_cycle_err_u2 = math.sqrt(sum(self.err_u2[N0:N1]) / N)
                    last_cycle_div2 = sum(self.div_u2[N0:N1]) / N
                    last_cycle_err_min2 = math.sqrt(min(self.err_u2[N0:N1]))
                    last_cycle_err_max2 = math.sqrt(max(self.err_u2[N0:N1]))

            self.err_u = [math.sqrt(i) for i in self.err_u]
            self.err_u2 = [math.sqrt(i) for i in self.err_u2]

            # report of error norm for individual time steps
            with open(self.str_dir_name + "/report_err.csv", 'w') as reportFile:
                report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
                report_writer.writerow(["name"] + ["time"] + self.time_list)
                report_writer.writerow([str_name] + ["corrected"] + self.err_u)
                if self.hasTentativeVel:
                    report_writer.writerow([str_name] + ["tentative"] + self.err_u2)

        # report of norm of div for individual time steps
        with open(self.str_dir_name + "/report_div.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow([str_name] + ["corrected"] + self.div_u)
            if self.hasTentativeVel:
                report_writer.writerow([str_name] + ["tentative"] + self.div_u2)

        # report without header
        with open(self.str_dir_name + "/report.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [mesh] + [factor] + [ttime] + [dt] + [
                    total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] + [avg_err_u] + [avg_err_u2] + [
                    last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] + [last_cycle_div2] + [last_cycle_err_min] + [
                    last_cycle_err_max] + [last_cycle_err_min2] + [last_cycle_err_max2])

        # report with header
        with open(self.str_dir_name + "/report_h.csv", 'w') as reportFile:
            report_writer = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            report_writer.writerow(
                ["problem"] + ["name"] + ["type"] + ["method"] + ["mesh_name"] + ["mesh"] + ["factor"] + ["time"] + ["dt"] + [
                    "timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"] + ["avg_err_u"] + [
                    "avg_err_u2"] + ["last_cycle_err_u"] + ["last_cycle_err_u2"] + ["last_cycle_div"] + ["last_cycle_div2"] + [
                    "last_cycle_err_min"] + ["last_cycle_err_max"] + ["last_cycle_err_min2"] + ["last_cycle_err_max2"])
            report_writer.writerow(
                ["pipe_test"] + [str_name] + [str_type] + [str_method] + [mesh_name] + [mesh] + [factor] + [ttime] + [dt] + [
                    total - self.time_erc] + [self.time_erc] + [total_err_u] + [total_err_u2] + [avg_err_u] + [avg_err_u2] + [
                    last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] + [last_cycle_div2] + [last_cycle_err_min] + [
                    last_cycle_err_max] + [last_cycle_err_min2] + [last_cycle_err_max2])

        # create file showing all was done well
        f = open(str_name + "_factor%4.2f_step_%dms_OK.report" % (factor, dt * 1000), "w")
        f.close()




