from __future__ import print_function
from dolfin import *
# from dolfin import assemble, interpolate, Expression, Function, DirichletBC, norm
# from dolfin.cpp.common import toc, mpi_comm_world
# from dolfin.cpp.io import HDF5File
# from dolfin.cpp.mesh import Mesh, MeshFunction
# from ufl import Measure, dx, cos, sin, Constant
from math import pi, sqrt

from problems import general_problem as gp
import womersleyBC


class Problem(object, gp.GeneralProblem):
    def __init__(self, args, tc, metadata):
        self.has_analytic_solution = True
        self.problem_code = 'WCYL'
        gp.GeneralProblem.__init__(self, args, tc, metadata)

        # TODO check if really used here
        self.tc.init_watch('assembleSol', 'Assembled analytic solution', True)
        self.tc.init_watch('analyticP', 'Analytic pressure', True)
        self.tc.init_watch('analyticVnorms', 'Computed analytic velocity norms', True)
        self.tc.init_watch('errorP', 'Computed pressure error', True)
        self.tc.init_watch('errorV', 'Computed velocity error', True)
        self.tc.init_watch('errorForce', 'Computed force error', True)
        self.tc.init_watch('errorVtest', 'Computed velocity error test', True)
        self.tc.init_watch('div', 'Computed and saved divergence', True)
        self.tc.init_watch('divNorm', 'Computed norm of divergence', True)

        self.name = 'womersley_cylinder'
        self.status_functional_str = 'last H1 velocity error'

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
        self.solution = None
        self.sol_p = None
        self.last_analytic_pressure_norm = None
        self.vel_normalization_factor = []
        self.pg_normalization_factor = []
        self.p_normalization_factor = []
        self.v_in = None
        self.area = None

        choose_note = {1.0: '', 0.1: 'nuL10', 0.01: 'nuL100', 10.0: 'nuH10'}
        self.precomputed_filename = args.mesh + choose_note[self.nu_factor]
        print('chosen filename for precomputed solution', self.precomputed_filename)

        # partial Bessel functions and coefficients
        self.bessel_parabolic = None
        self.bessel_real = []
        self.bessel_complex = []
        self.coefs_exp = [-8, -6, -4, -2, 2, 4, 6, 8]

        self.analytic_v_norm_L2 = None
        self.analytic_v_norm_H1 = None
        self.analytic_v_norm_H1w = None
        # lists
        # dictionary of data lists {list, name, abbreviation, add scaled row to report}
        # normalisation coefficients (time-independent) are added to some lists to be used in normalized data series
        #   coefficients are lists (updated during initialisation, so we cannot use float type)
        #   coefs are equal to average of respective value of analytic solution
        # norm lists (time-dependent normalisation coefficients) are added to some lists to be used in relative data
        #  series (to remove natural pulsation of error due to change in volume flow rate)
        # slist - lists for cycle-averaged values
        # L2(0) means L2 difference of pressures taken with zero average
        self.listDict = {
            'u_L2': {'list': [], 'name': 'corrected velocity L2 error', 'abrev': 'CE_L2', 'scale': self.factor,
                     'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
            'u2L2': {'list': [], 'name': 'tentative velocity L2 error', 'abrev': 'TE_L2', 'scale': self.factor,
                     'relative': 'av_norm_L2', 'slist': [], 'norm': self.vel_normalization_factor},
            'u_L2test': {'list': [], 'name': 'test corrected L2 velocity error', 'abrev': 'TestCE_L2', 'scale': self.factor},
            'u2L2test': {'list': [], 'name': 'test tentative L2 velocity error', 'abrev': 'TestTE_L2', 'scale': self.factor},
            'u_H1': {'list': [], 'name': 'corrected velocity H1 error', 'abrev': 'CE_H1', 'scale': self.factor,
                     'relative': 'av_norm_H1', 'slist': []},
            'u_H1w': {'list': [], 'name': 'corrected velocity H1 error on wall', 'abrev': 'CE_H1w', 'scale': self.factor,
                     'relative': 'av_norm_H1w', 'slist': []},
            'u2H1': {'list': [], 'name': 'tentative velocity H1 error', 'abrev': 'TE_H1', 'scale': self.factor,
                     'relative': 'av_norm_H1', 'slist': []},
            'u2H1w': {'list': [], 'name': 'tentative velocity H1 error on wall', 'abrev': 'TE_H1w', 'scale': self.factor,
                     'relative': 'av_norm_H1w', 'slist': []},
            'u_H1test': {'list': [], 'name': 'test corrected H1 velocity error', 'abrev': 'TestCE_H1', 'scale': self.factor},
            'u2H1test': {'list': [], 'name': 'test tentative H1 velocity error', 'abrev': 'TestTE_H1', 'scale': self.factor},
            'd': {'list': [], 'name': 'corrected velocity L2 divergence', 'abrev': 'DC', 'scale': self.factor, 'slist': []},
            'd2': {'list': [], 'name': 'tentative velocity L2 divergence', 'abrev': 'DT', 'scale': self.factor, 'slist': []},
            'apg': {'list': [], 'name': 'analytic pressure gradient', 'abrev': 'APG', 'scale': self.factor,
                    'norm': self.pg_normalization_factor},
            'av_norm_L2': {'list': [], 'name': 'analytic velocity L2 norm', 'abrev': 'AVN_L2'},
            'av_norm_H1': {'list': [], 'name': 'analytic velocity H1 norm', 'abrev': 'AVN_H1'},
            'av_norm_H1w': {'list': [], 'name': 'analytic velocity H1 norm on wall', 'abrev': 'AVN_H1w'},
            'a_force_wall': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AF'},
            'a_force_wall_normal': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AFN'},
            'a_force_wall_shear': {'list': [], 'name': 'analytic force on wall', 'abrev': 'AFS'},
            'force_wall': {'list': [], 'name': 'force error on wall', 'abrev': 'FE',
                           'relative': 'a_force_wall', 'slist': []},
            'force_wall_normal': {'list': [], 'name': 'normal force error on wall', 'abrev': 'FNE',
                                  'relative': 'a_force_wall', 'slist': []},
            'force_wall_shear': {'list': [], 'name': 'shear force error on wall', 'abrev': 'FSE',
                                 'relative': 'a_force_wall', 'slist': []},
            'ap_norm': {'list': [], 'name': 'analytic pressure norm', 'abrev': 'APN'},
            'p': {'list': [], 'name': 'pressure L2(0) error', 'abrev': 'PE', 'scale': self.factor, 'slist': [],
                  'norm': self.p_normalization_factor},
            'pg': {'list': [], 'name': 'computed pressure gradient', 'abrev': 'PG', 'scale': self.factor,
                   'norm': self.pg_normalization_factor},
            'pgE': {'list': [], 'name': 'computed pressure gradient error', 'abrev': 'PGE', 'scale': self.factor,
                    'norm': self.pg_normalization_factor, 'slist': []},
            'pgEA': {'list': [], 'name': 'computed absolute pressure gradient error', 'abrev': 'PGEA',
                     'scale': self.factor, 'norm': self.pg_normalization_factor},
            'p2': {'list': [], 'name': 'pressure tent L2(0) error', 'abrev': 'PTE', 'scale': self.factor,
                   'slist': [], 'norm': self.p_normalization_factor},
            'pg2': {'list': [], 'name': 'computed pressure tent gradient', 'abrev': 'PTG', 'scale': self.factor,
                    'norm': self.pg_normalization_factor},
            'pgE2': {'list': [], 'name': 'computed tent pressure tent gradient error', 'abrev': 'PTGE',
                     'scale': self.factor, 'norm': self.pg_normalization_factor, 'slist': []},
            'pgEA2': {'list': [], 'name': 'computed absolute pressure tent gradient error',
                      'abrev': 'PTGEA', 'scale': self.factor, 'norm': self.pg_normalization_factor}
            }

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
        super(Problem, self).initialize(V, Q, PS)

        print("Problem type: " + self.type)
        print("Velocity scale factor = %4.2f" % self.factor)
        reynolds = 728.761 * self.factor  # TODO modify by nu_factor
        print("Computing with Re = %f" % reynolds)

        self.v_in = Function(V)
        print('Initializing error control')
        self.load_precomputed_bessel_functions(PS)

        # set constants for
        self.area = assemble(interpolate(Expression("1.0"), Q) * self.dsIn)  # inflow area

        # if self.doErrControl and self.isSteadyFlow:  # NT Steady case
        #     temp = toc()
        #     self.solution = interpolate(
        #         Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=self.factor), solution_space)
        #     print("Prepared analytic solution. Time: %f" % (toc() - temp))

        # self.pg_normalization_factor.append(womersleyBC.average_analytic_pressure_grad(self.factor))
        self.p_normalization_factor.append(norm(
            interpolate(womersleyBC.average_analytic_pressure_expr(self.factor), self.pSpace), norm_type='L2'))
        self.vel_normalization_factor.append(norm(
            interpolate(womersleyBC.average_analytic_velocity_expr(self.factor), self.vSpace), norm_type='L2'))
        # print('Normalisation factors (vel, p, pg):', self.vel_normalization_factor[0], self.p_normalization_factor[0],
        #       self.pg_normalization_factor[0])

    def get_boundary_conditions(self, V, Q, use_pressure_BC):
        # boundary parts: 1 walls, 2 inflow, 3 outflow
        noSlip = Constant((0.0, 0.0, 0.0))    # IMP fails when using direct import from UFL
        # if self.type == "steady":   # NT Steady
        #     self.v_in = Expression(("0.0", "0.0",
        #                        "(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):\
        #                        (factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),
        #                       t=0, factor=self.factor)

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
        self.time_list.append(self.actual_time)
        if self.actual_time > 0.5 and int(round(self.actual_time * 1000)) % 1000 == 0:
            self.isWholeSecond = True
            seconds = int(round(self.actual_time))
            self.second_list.append(seconds)
            self.N1 = seconds*self.stepsInSecond
            self.N0 = (seconds-1)*self.stepsInSecond
        else:
            self.isWholeSecond = False
        if not self.type == 'steady':
            self.solution = self.assemble_solution(self.actual_time)

            # Update boundary condition
            self.tc.start('updateBC')
            # if self.type == "steady":  NT Steady
            #     self.v_in.t = self.actual_time
            # else:
            self.v_in.assign(self.solution)
            self.tc.end('updateBC')

            self.tc.start('analyticVnorms')
            self.analytic_v_norm_L2 = norm(self.solution, norm_type='L2')
            self.analytic_v_norm_H1 = norm(self.solution, norm_type='H1')
            self.analytic_v_norm_H1w = sqrt(assemble((inner(grad(self.solution), grad(self.solution)) +
                                                      inner(self.solution, self.solution)) * self.dsWall))
            self.listDict['av_norm_L2']['list'].append(self.analytic_v_norm_L2)
            self.listDict['av_norm_H1']['list'].append(self.analytic_v_norm_H1)
            self.listDict['av_norm_H1w']['list'].append(self.analytic_v_norm_H1w)
            self.tc.end('analyticVnorms')

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

    def save_vel(self, is_tent, field, t):
        super(Problem, self).save_vel(is_tent, field, t)
        if self.doSaveDiff and t > 0.00001:  # NT maybe not needed, if solution is initialized before first save...
            self.vFunction.assign((1.0 / self.vel_normalization_factor[0]) * (field - self.solution))
            self.fileDict['u2D' if is_tent else 'uD']['file'] << self.vFunction

    def compute_err(self, is_tent, velocity, t):
        if self.doErrControl:
            er_list_L2 = self.listDict['u2L2' if is_tent else 'u_L2']['list']
            er_list_H1 = self.listDict['u2H1' if is_tent else 'u_H1']['list']
            er_list_H1w = self.listDict['u2H1w' if is_tent else 'u_H1w']['list']
            if self.testErrControl:
                er_list_test_H1 = self.listDict['u2H1test' if is_tent else 'u_H1test']['list']
                er_list_test_L2 = self.listDict['u2L2test' if is_tent else 'u_L2test']['list']
            # if self.isSteadyFlow: NT Steady
            #     if self.testErrControl:
            #         self.tc.start('errorVtest')
            #         er_list_test_L2.append(errornorm(velocity, self.solution, norm_type='L2', degree_rise=0))
            #         er_list_test_H1.append(errornorm(velocity, self.solution, norm_type='H1', degree_rise=0))
            #         self.tc.end('errorVtest')
            #     self.tc.start('errorV')
            #     er_list_L2.append(assemble(inner(velocity - self.solution, velocity - self.solution) * dx))  # faster
            #     self.tc.end('errorV')
            # else:
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
            self.last_error = errorH1 / self.analytic_v_norm_H1
            self.last_status_functional = self.last_error
            print("  Relative H1 error in velocity = ", self.last_error)
            er_list_L2.append(errorL2)
            er_list_H1.append(errorH1)
            errorH1wall = sqrt(assemble((inner(grad(velocity - self.solution), grad(velocity - self.solution)) +
                                         inner(velocity - self.solution, velocity - self.solution)) * self.dsWall))
            er_list_H1w.append(errorH1wall)
            print('  Relative H1wall error:', errorH1wall / self.analytic_v_norm_H1w)
            self.tc.end('errorV')
            if self.isWholeSecond:
                self.listDict['u2L2' if is_tent else 'u_L2']['slist'].append(
                    sqrt(sum([i*i for i in er_list_L2[self.N0:self.N1]])/self.stepsInSecond))
                self.listDict['u2H1' if is_tent else 'u_H1']['slist'].append(
                    sqrt(sum([i*i for i in er_list_H1[self.N0:self.N1]])/self.stepsInSecond))
                self.listDict['u2H1w' if is_tent else 'u_H1w']['slist'].append(
                    sqrt(sum([i*i for i in er_list_H1w[self.N0:self.N1]])/self.stepsInSecond))
            # stopping criteria
            if self.last_error > self.divergence_treshold:
                self.report_divergence(t)

    def save_pressure(self, is_tent, pressure):
        super(Problem, self).save_pressure(is_tent, pressure)
        self.tc.start('computePG')
        # Report pressure gradient
        p_in = assemble((1.0/self.area) * pressure * self.dsIn)
        p_out = assemble((1.0/self.area) * pressure * self.dsOut)
        computed_gradient = (p_out - p_in)/20.0
        # 20.0 is a length of a pipe NT should depend on mesh length (implement throuhg metadata or function of mesh)
        self.tc.end('computePG')
        self.tc.start('analyticP')
        analytic_gradient = womersleyBC.analytic_pressure_grad(self.factor, self.actual_time)
        analytic_pressure = womersleyBC.analytic_pressure(self.factor, self.actual_time)
        self.sol_p = interpolate(analytic_pressure, self.pSpace)  # NT move to update_time
        if not is_tent:
            self.last_analytic_pressure_norm = norm(self.sol_p, norm_type='L2')
            self.listDict['ap_norm']['list'].append(self.last_analytic_pressure_norm)
        self.tc.end('analyticP')
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
        if self.doSaveDiff:
            sol_pg_expr = Expression(("0", "0", "pg"), pg=analytic_gradient / self.pg_normalization_factor[0])
            # sol_pg = interpolate(sol_pg_expr, self.pgSpace)
            # plot(sol_p, title="sol")
            # plot(pressure, title="p")
            # plot(pressure - sol_p, interactive=True, title="diff")
            # exit()
            self.pFunction.assign(pressure-self.sol_p)
            self.fileDict['p2D' if is_tent else 'pD']['file'] << self.pFunction
            # self.pgFunction.assign(pg-sol_pg)
            # self.fileDict['pg2D' if is_tent else 'pgD'][0] << self.pgFunction

