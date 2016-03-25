from problems import general_problem as gp
from dolfin import *


class Problem(gp.GeneralProblem):
    def __init__(self, args, tc):
        gp.GeneralProblem.__init__(self, args, tc)

        # TODO check
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
        self.tc.init_watch('status', 'Reported status.', True)

        self.name = 'womersley_cylinder'

        # input parameters
        self.type = args.type
        self.factor = args.factor

        # fixed parameters (used in analytic solution and in BC)
        self.nu = 3.71 * args.nu # kinematic viscosity
        self.R = 5.0  # cylinder radius

        # Import gmsh mesh
        self.mesh = Mesh("meshes/" + args.mesh + ".xml")
        self.cell_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_physical_region.xml")
        self.facet_function = MeshFunction("size_t", self.mesh, "meshes/" + args.mesh + "_facet_region.xml")
        self.dsIn = Measure("ds", subdomain_id=2, subdomain_data=self.facet_function)
        self.dsOut = Measure("ds", subdomain_id=3, subdomain_data=self.facet_function)
        self.dsWall = Measure("ds", subdomain_id=1, subdomain_data=self.facet_function)
        # QQ rm.dsWall = dsWall
        print("Mesh name: ", args.mesh, "    ", self.mesh)
        print("Mesh norm max: ", self.mesh.hmax())
        print("Mesh norm min: ", self.mesh.hmin())

        self.actual_time = None
        self.solutionSpace = None
        self.solution = None
        self.v_in = None

        choose_note = {1.0: '', 0.1: 'nuL10', 0.01: 'nuL100', 10.0: 'nuH10'}
        self.precomputed_filename = args.mesh + choose_note[self.nu_factor]
        print('chosen filename for precomputed solution', self.precomputed_filename)

        # partial Bessel functions and coefficients
        self.bessel_parabolic = None
        self.bessel_real = []
        self.bessel_complex = []
        self.coefs_exp = [-8, -6, -4, -2, 2, 4, 6, 8]

    def __str__(self):
        return 'womersley flow in cylinder'

    @staticmethod
    def setup_parser_options(parser):
        gp.GeneralProblem.setup_parser_options(parser)
        # TODO split steady and unsteady problem (move steady to steady_cylinder)
        # TODO split type of flow and initial condition type
        # QQ pulsePrec?
        # IFNEED smooth initial u0 v_in incompatibility via modification of v_in (options normal, smoothed)
        parser.add_argument('-T', '--type', help='Flow type', choices=['steady', 'pulse0', 'pulsePrec'], default='pulse0')
        #   steady - parabolic profile (0.5 s onset)
        # Womersley profile (1 s period)
        #   pulse0 - u(0)=0
        #   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
        parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)

    def initialize(self, V, Q, PS):
        print("Problem type: " + self.type)
        print("Velocity scale factor = %4.2f" % self.factor)
        reynolds = 728.761 * self.factor  # TODO modify by nu_factor
        print("Computing with Re = %f" % reynolds)

        self.v_in = Function(V)
        self.solutionSpace = V
        self.load_precomputed_bessel_functions(PS)

        # set constants for
        area = assemble(interpolate(Expression("1.0"), Q) * self.dsIn)  # inflow area
        volume = assemble(interpolate(Expression("1.0"), Q) * dx)

    def get_boundary_conditions(self, V, Q, use_pressure_BC):
        # boundary parts: 1 walls, 2 inflow, 3 outflow
        noSlip = Constant((0.0, 0.0, 0.0))
        if self.type == "steady":
            self.v_in = Expression(("0.0", "0.0",
                               "(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):\
                               (factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),
                              t=0, factor=self.factor)

        # Boundary conditions
        bc0 = DirichletBC(V, noSlip, self.facet_function, 1)
        inflow = DirichletBC(V, self.v_in, self.facet_function, 2)
        bcu = [inflow, bc0]
        bcp = []
        if use_pressure_BC:
            outflow = DirichletBC(Q, Constant(0.0), self.facet_function, 3)
            bcp = [outflow]
        return bcu, bcp

    def get_initial_conditions(self, V, Q):
        v0 = Function(V)
        p0 = Function(Q)
        # if self.type == "pulsePrec":  # QQ implement?
        #     assign(u0, u_prec)
        #     assign(p0, p_prec)
        # TODO analytic u0

        return v0, p0

    def update_time(self, actual_time):
        self.actual_time = actual_time
        # self.write_status_file(self.actual_time) TODO implement for general_solver
        # self.time_list.append(self.actual_time)
        # if self.actual_time > 0.5 and int(round(self.actual_time * 1000)) % 1000 == 0:
        #     self.isWholeSecond = True
        #     seconds = int(round(self.actual_time))
        #     self.second_list.append(seconds)
        #     self.N1 = seconds*self.stepsInSecond
        #     self.N0 = (seconds-1)*self.stepsInSecond
        # else:
        #     self.isWholeSecond = False
        if not self.type == 'steady':
            self.solution = self.assemble_solution(self.actual_time)

            # Update boundary condition
            self.tc.start('updateBC')
            if self.type == "steady":
                self.v_in.t = self.actual_time
            else:
                self.v_in.assign(self.solution)
            self.tc.end('updateBC')

            # self.tc.start('analyticVnorms')
            # self.analytic_v_norm_L2 = norm(self.solution, norm_type='L2')
            # self.analytic_v_norm_H1 = norm(self.solution, norm_type='H1')
            # self.analytic_v_norm_H1w = sqrt(assemble((inner(grad(self.solution), grad(self.solution)) +
            #                                           inner(self.solution, self.solution)) * self.dsWall))
            # self.listDict['av_norm_L2']['list'].append(self.analytic_v_norm_L2)
            # self.listDict['av_norm_H1']['list'].append(self.analytic_v_norm_H1)
            # self.listDict['av_norm_H1w']['list'].append(self.analytic_v_norm_H1w)
            # self.tc.end('analyticVnorms')

    def assemble_solution(self, t):  # returns Womersley sol for time t
        if self.tc is not None:
            self.tc.start('assembleSol')
        sol = Function(self.solutionSpace)
        dofs2 = self.solutionSpace.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
        sol.assign(Constant(("0.0", "0.0", "0.0")))
        sol.vector()[dofs2] += self.factor * self.bessel_parabolic.vector().array()  # parabolic part of sol
        for idx in range(8):  # add modes of Womersley sol
            sol.vector()[dofs2] += self.factor * cos(self.coefs_exp[idx] * pi * t) * self.bessel_real[idx].vector().array()
            sol.vector()[dofs2] += self.factor * -sin(self.coefs_exp[idx] * pi * t) * self.bessel_complex[idx].vector().array()
        if self.tc is not None:
            self.tc.end('assembleSol')
        return sol

    # load precomputed Bessel functions
    def load_precomputed_bessel_functions(self, PS):
        f = HDF5File(mpi_comm_world(), 'precomputed/precomputed_' + self.precomputed_filename + '.hdf5', 'r')
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

