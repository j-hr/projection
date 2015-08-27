from __future__ import print_function

__author__ = 'jh'

from dolfin import *
import os, traceback


class ResultsManager:
    def __init__(self):

        self.c0_prec = None
        self.coefs_r_prec = []
        self.coefs_i_prec = []

        self.str_dir_name = 0
        self.doSave = None
        self.hasTentativeVel = False

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

    # load precomputed Bessel functions=================================================================================
    def load_precomputed_bessel_functions(self, meshName, PS):
        f = HDF5File(mpi_comm_world(), 'precomputed/precomputed_'+meshName+'.hdf5', 'r')
        temp = toc()
        fce = Function(PS)
        f.read(fce,"parab")
        global c0_prec
        c0_prec = Function(fce)
        for i in range(8):
            f.read(fce, "real%d" % i)
            self.coefs_r_prec.append(Function(fce))
            f.read(fce, "imag%d" % i)
            self.coefs_i_prec.append(Function(fce))
            # plot(coefs_r_prec[i], title="coefs_r_prec", interactive=True) # reasonable values
            # plot(coefs_i_prec[i], title="coefs_i_prec", interactive=True) # reasonable values
        # plot(c0_prec,title="c0_prec",interactive=True) # reasonable values
        print("Loaded partial solution functions. Time: %f" % (toc() - temp))

    def prepare_analytic_solution(self, str_type, factor, V):
        temp = toc()
        if str_type == "steady":
            global solution
            solution = interpolate(
                Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=factor), V)
            print("Prepared analytic solution. Time: %f" % (toc() - temp))
        elif (str_type == "pulse0") or (str_type == "pulsePrec"):
            def assembleSolution(self, t):  # returns Womersley sol for time t
                tmp = toc()
                sol = Function(V)
                dofs2 = V.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
                sol.assign(Constant(("0.0", "0.0", "0.0")))
                sol.vector()[dofs2] = factor * c0_prec.vector().array()  # parabolic part of sol
                self.coefs_exp = [-8, 8, 6, -6, -4, 4, 2, -2]
                for idx in range(8):  # add modes of Womersley sol
                    sol.vector()[dofs2] += factor * cos(self.coefs_exp[idx] * pi * t) * self.coefs_r_prec[idx].vector().array()
                    sol.vector()[dofs2] += factor * -sin(self.coefs_exp[idx] * pi * t) * self.coefs_i_prec[idx].vector().array()
                print("Assembled analytic solution. Time: %f" % (toc() - tmp))
                return sol

            # plot(assembleSolution(0.0), mode = "glyphs", title="sol")
            # interactive()
            # exit()
            # save solution
            # f=File("sol.xdmf")
            # t = dt
            # s= Function(V)
            # while t < Time + DOLFIN_EPS:
            # print("t = ", t)
            # s.assign(assembleSolution(t))
            # f << s
            # t+=dt
            # exit()

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

    def initialize_output(self, velocity_space, mesh):
        print('Initializing output')
        # create directory, needed because of using "with open(..." construction later
        if not os.path.exists(self.str_dir_name):
            os.mkdir(self.str_dir_name)
        if self.doSave:
            self.vel = Function(velocity_space)
            self.D = FunctionSpace(mesh, "Lagrange", 1)
            self.divFunction = Function(self.D)
            self.initialize_xdmf_files()

    # method for saving divergence (ensuring, that it will be one time line in ParaView)
    def save_div(self, isTent, field):
        div_file = self.d2File if isTent else self.dFile
        tmp = toc()
        self.divFunction.assign(project(div(field), self.D))
        div_file << self.divFunction
        print("Computed and saved divergence. Time: %f" % (toc() - tmp))

    # method for saving velocity (ensuring, that it will be one time line in ParaView)
    def save_vel(self, isTent, field):
        vel_file = self.u2File if isTent else self.uFile
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





