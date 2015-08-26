from __future__ import print_function
from dolfin import *
import math
import csv
import sys
import os
from sympy import I, re, im, sqrt, exp, symbols, lambdify, besselj
from scipy.special import jv
import traceback

# TODO ? load precomputed solutions from files (to save hours of cputime)

# TODO if method == ... change to if hasTentative
# TODO another projection methods (MiroK)

# Issues
# TODO cyl3 cannot be solved directly, pulsePrec also fails. try mumps, then paralelize
# mesh.hmax() returns strange values

# Notes
# characteristic time for onset ~~ length of pipe/speed of fastest particle = 20(mm /factor*1081(mm/s) ~~  0.02 s/factor
# characteristic time for dt ~~ hmax/speed of fastest particle = hmax/(factor*1081(mm/s))
#   h = cubeRoot(volume/number of cells):
#   c1: 829 cells => h = 1.23 mm => 1.1 ms/factor
#   c2: 6632 cells => h = 0.62 mm => 0.57 ms/factor
#   c3: 53056 cells => h = 0.31 mm => 0.28 ms/factor

tic()
# Resolve input arguments===============================================================================================
print(sys.argv)
nargs = 9
arguments = "method type mesh_name T dt(minimal: 0.001) factor \
error_control_start_time(-1 for default starting time, 0 for no errc.) save_results(save/noSave) name"
if len(sys.argv) != nargs + 1:
    exit("Wrong number of arguments. Should be: %d (%s)" % (nargs, arguments))

# name used for result folders and report files
str_name = sys.argv[9]

# save mode
#   save: create .xdmf files with velocity, pressure, divergence
#   noSave: do not create .xdmf files with velocity, pressure, divergence
doSave = False
if sys.argv[8] == 'save':
    doSave = True
    print('Saving solution ON.')
elif sys.argv[8] == 'noSave':
    doSave = False
    print('Saving solution OFF.')
else:
    exit('Wrong parameter save_results.')

# choose a method: direct, chorinExpl
str_method = sys.argv[1]
print("Method:       " + str_method)
# choose type of flow:
#   steady - parabolic profile (0.5 s onset)
# Womersley profile (1 s period)
#   pulse0 - u(0)=0
#   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
str_type = sys.argv[2]
print("Problem type: " + str_type)

if sys.argv[7] == "0":
    doErrControl = False
    print("Error control omitted")
else:
    doErrControl = True
    if sys.argv[7] == "-1":
        measure_time = 0.5 if str_type == "steady" else 1  # maybe change
    else:
        measure_time = float(sys.argv[7])
    print("Error control from:       %4.2f s" % measure_time)

# Set parameter values
dt = float(sys.argv[5])
time = float(sys.argv[4])
print("Time:         %1.0f s\ndt:           %d ms" % (time, 1000 * dt))
factor = float(sys.argv[6])  # default: 1.0
print("Velocity scale factor = %4.2f" % factor)
reynolds = 728.761 * factor
print("Computing with Re = %f" % reynolds)

# Import gmsh mesh 
meshName = sys.argv[3]
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")
print("Mesh norm: ", mesh.hmax(), "    ", mesh)
# ======================================================================================================================
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

# fixed parameters (used in analytic solution)
nu = 3.71  # kinematic viscosity
R = 5.0  # cylinder radius

# MOVE#==precomputation of Bessel functions=============================================================================
if doErrControl:
    if str_type == "pulse0" or str_type == "pulsePrec":
        temp = toc()
        coefs_mult = [(-11.799 + 0.60076 * I), (-11.799 - 0.60076 * I), (-26.3758 - 4.65265 * I),
                      (-26.3758 + 4.65265 * I), (-51.6771 + 27.3133 * I), (-51.6771 - 27.3133 * I),
                      (-33.1594 - 95.2423 * I), (-33.1594 + 95.2423 * I)]
        coefs_bes_mult = [(0.000735686 - 0.000528035 * I), (0.000735686 + 0.000528035 * I),
                          -(0.000814244 - 0.00277126 * I), -(0.000814244 + 0.00277126 * I),
                          -(0.0110653 - 0.00200668 * I), -(0.0110653 + 0.00200668 * I), (0.0314408 - 0.0549981 * I),
                          (0.0314408 + 0.0549981 * I)]
        coefs_bes = [(1.84042 + 1.84042 * I), (1.84042 - 1.84042 * I), (1.59385 - 1.59385 * I), (1.59385 + 1.59385 * I),
                     (1.30138 + 1.30138 * I), (1.30138 - 1.30138 * I), (0.920212 - 0.920212 * I),
                     (0.920212 + 0.920212 * I)]
        coefs_exp = [-8, 8, 6, -6, -4, 4, 2, -2]
        coefs_r_prec = []  # these will be functions in PS
        coefs_i_prec = []  # these will be functions in PS
        c0ex = Expression("factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))", factor=factor)
        c0_prec = (interpolate(c0ex, PS))
        for i in range(8):
            r = symbols('r')
            besRe = re(factor * coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
            besIm = im(factor * coefs_mult[i] * (coefs_bes_mult[i] * besselj(0, r * coefs_bes[i]) + 1))
            besRe_lambda = lambdify(r, besRe, ['numpy', {'besselj': jv}])
            besIm_lambda = lambdify(r, besIm, ['numpy', {'besselj': jv}])


            class PartialReSolution(Expression):
                def eval(self, value, x):
                    rad = float(sqrt(x[0] * x[0] + x[1] * x[
                        1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                    value[0] = 0 if near(rad, R) else besRe_lambda(rad)  # do not evaluate on boundaries, it's 0
                    # print(value) gives reasonable values


            expr = PartialReSolution()
            coefs_r_prec.append(interpolate(expr, PS))


            class PartialImSolution(Expression):
                def eval(self, value, x):
                    rad = float(sqrt(x[0] * x[0] + x[1] * x[
                        1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                    value = 0 if near(rad, R) else besIm_lambda(rad)  # do not evaluate on boundaries, it's 0


            expr = PartialImSolution()
            coefs_i_prec.append(interpolate(expr, PS))
        # plot(coefs_r_prec[2],title="coefs_r_prec") #reasonable values
        print("Precomputed partial solution functions. Time: %f" % (toc() - temp))

# MOVE#==Boundary Conditions================================================================================
# boundary parts: 1 walls, 2 inflow, 3 outflow
noSlip = Constant((0.0, 0.0, 0.0))
if str_type == "steady":
    v_in = Expression(("0.0", "0.0",
                       "(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):\
                       (factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),
                      t=0, factor=factor)
elif str_type == "pulse0" or str_type == "pulsePrec":
    r, t = symbols('r t')
    u = factor * (-43.2592 * r ** 2 +
                  (-11.799 + 0.60076 * I) * ((0.000735686 - 0.000528035 * I)
                                             * besselj(0, r * (1.84042 + 1.84042 * I)) + 1) * exp(-8 * I * pi * t) +
                  (-11.799 - 0.60076 * I) * ((0.000735686 + 0.000528035 * I)
                                             * besselj(0, r * (1.84042 - 1.84042 * I)) + 1) * exp(8 * I * pi * t) +
                  (-26.3758 - 4.65265 * I) * (-(0.000814244 - 0.00277126 * I)
                                              * besselj(0, r * (1.59385 - 1.59385 * I)) + 1) * exp(6 * I * pi * t) +
                  (-26.3758 + 4.65265 * I) * (-(0.000814244 + 0.00277126 * I)
                                              * besselj(0, r * (1.59385 + 1.59385 * I)) + 1) * exp(-6 * I * pi * t) +
                  (-51.6771 + 27.3133 * I) * (-(0.0110653 - 0.00200668 * I)
                                              * besselj(0, r * (1.30138 + 1.30138 * I)) + 1) * exp(-4 * I * pi * t) +
                  (-51.6771 - 27.3133 * I) * (-(0.0110653 + 0.00200668 * I)
                                              * besselj(0, r * (1.30138 - 1.30138 * I)) + 1) * exp(4 * I * pi * t) +
                  (-33.1594 - 95.2423 * I) * ((0.0314408 - 0.0549981 * I)
                                              * besselj(0, r * (0.920212 - 0.920212 * I)) + 1) * exp(2 * I * pi * t) +
                  (-33.1594 + 95.2423 * I) * ((0.0314408 + 0.0549981 * I)
                                              * besselj(0, r * (0.920212 + 0.920212 * I)) + 1) * exp(
                      -2 * I * pi * t) + 1081.48)
    # how this works?
    u_lambda = lambdify([r, t], u, ['numpy', {'besselj': jv}])


    class WomersleyProfile(Expression):
        def eval(self, value, x):
            rad = float(sqrt(x[0] * x[0] + x[1] * x[
                1]))  # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
            value[0] = 0
            value[1] = 0
            value[2] = 0 if near(rad, R) else re(u_lambda(rad, t))  # do not evaluate on boundaries, it's 0

        def value_shape(self):
            return (3,)


    v_in = WomersleyProfile()
# MOVE#==Initial Conditions================================================================================
if str_type == "pulsePrec":  # computes initial velocity as a solution of steady Stokes problem with input velocity v_in
    temp = toc()
    begin("Computing initial velocity")
    t = 0  # used in v_in

    # Define function spaces (Taylor-Hood)
    W = MixedFunctionSpace([V, Q])
    bc0 = DirichletBC(W.sub(0), noSlip, facet_function, 1)
    inflow = DirichletBC(W.sub(0), v_in, facet_function, 2)
    # Collect boundary conditions
    bcu = [inflow, bc0]
    # Define unknown and test function(s) NS
    v, q = TestFunctions(W)
    u, p = TrialFunctions(W)
    w = Function(W)
    # Define fields
    n = FacetNormal(mesh)
    I = Identity(u.geometric_dimension())  # Identity tensor
    x = SpatialCoordinate(mesh)


    # Define steady part of the equation
    def T(u):
        return -p * I + 2.0 * nu * sym(grad(u))


    # Define variational forms
    F = (inner(T(u), grad(v)) - q * div(u)) * dx
    solve(lhs(F) == rhs(F), w, bcu)  # why this way?
    # Extract solutions:
    (u_prec, p_prec) = w.split()
    print("Computed initial velocity. Time:%f" % (toc() - temp))
    end()

    # plot(u_prec, mode = "glyphs", title="steady solution", interactive=True)
    # exit()

# MOVE#==Analytic solution====================================================================================
if doErrControl:
    temp = toc()
    if str_type == "steady":
        global solution
        solution = interpolate(
            Expression(("0.0", "0.0", "factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"), factor=factor), V)
        print("Prepared analytic solution. Time: %f" % (toc() - temp))
    elif (str_type == "pulse0") or (str_type == "pulsePrec"):
        def assembleSolution(t):  # returns Womersley sol for time t
            tmp = toc()
            sol = Function(V)
            dofs2 = V.sub(2).dofmap().dofs()  # gives field of indices corresponding to z axis
            sol.assign(Constant(("0.0", "0.0", "0.0")))
            sol.vector()[dofs2] = c0_prec.vector()  # parabolic part of sol
            for idx in range(8):  # add modes of Womersley sol
                sol.vector()[dofs2] += cos(coefs_exp[idx] * pi * t) * coefs_r_prec[idx].vector().array()
                sol.vector()[dofs2] += -sin(coefs_exp[idx] * pi * t) * coefs_i_prec[idx].vector().array()
            print("Assembled analytic solution. Time: %f" % (toc() - tmp))
            return sol
            # plot(assembleSolution(0.2), mode = "glyphs", title="sol")
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

# Output settings=======================================================================================================
# Create files for storing solution
str_dir_name = str_name + "results_" + str_type + "_" + str_method + "_" + meshName + "_factor%4.2f_%ds_%dms" % (
    factor, time, dt * 1000)
if not os.path.exists(str_dir_name):
    os.mkdir(str_dir_name)
if doSave:
    uFile = XDMFFile(mpi_comm_world(), str_dir_name + "/velocity.xdmf")
    # saves lots of space (for use with static mesh)
    uFile.parameters['rewrite_function_mesh'] = False
    dFile = XDMFFile(mpi_comm_world(), str_dir_name + "/divergence.xdmf")  # maybe just compute norm
    dFile.parameters['rewrite_function_mesh'] = False
    pFile = XDMFFile(mpi_comm_world(), str_dir_name + "/pressure.xdmf")
    pFile.parameters['rewrite_function_mesh'] = False
    if str_method == "chorinExpl":
        u2File = XDMFFile(mpi_comm_world(), str_dir_name + "/velocity_tent.xdmf")
        u2File.parameters['rewrite_function_mesh'] = False
        d2File = XDMFFile(mpi_comm_world(), str_dir_name + "/div_tent.xdmf")  # maybe just compute norm
        d2File.parameters['rewrite_function_mesh'] = False

# method for saving divergence (ensuring, that it will be one time line in ParaView)
D = FunctionSpace(mesh, "Lagrange", 1)
divFunction = Function(D)


def save_div(field, divFile):
    tmp = toc()
    divFunction.assign(project(div(field), D))
    divFile << divFunction
    print("Computed and saved divergence. Time: %f" % (toc() - tmp))


# method for saving velocity (ensuring, that it will be one time line in ParaView)
vel = Function(V)


def save_vel(field, velFile):
    tmp = toc()
    vel.assign(field)
    velFile << vel
    print("Saved solution. Time: %f" % (toc() - tmp))


div_u = []
div_u2 = []


def compute_div(div_list, velocity):
    tmp = toc()
    div_list.append(norm(velocity, 'Hdiv0'))
    print("Computed norm of divergence. Time: %f" % (toc() - tmp))


def report_fail():
    print("Runtime error:", sys.exc_info()[1])
    print("Traceback:")
    traceback.print_tb(sys.exc_info()[2])
    f = open(str_name + "_factor%4.2f_step_%dms_failed_at_%5.3f.report" % (factor, dt * 1000, t), "w")
    f.write(traceback.format_exc())
    f.close()


# ==Error control====================================================================================
time_erc = 0  # total time spent on measuring error
time_list = []  # list of times, when error is  measured (used in report)
err_u = []
err_u2 = []
if doErrControl:

    def compute_err(er_list, velocity, t):
        if len(time_list) == 0 or (time_list[-1] < round(t, 3)):  # add only once per time step
            time_list.append(round(t, 3))  # round time step to 0.001
        tmp = toc()
        if str_type == "steady":
            # er_list.append(pow(errornorm(velocity, solution, norm_type='l2', degree_rise=0),2)) # slower, reliable
            er_list.append(assemble(inner(velocity - solution, velocity - solution) * dx))  # faster
        elif (str_type == "pulse0") or (str_type == "pulsePrec"):
            # er_list.append(pow(errornorm(velocity, assembleSolution(t), norm_type='l2', degree_rise=0),2))
            sol = assembleSolution(t)  # name must be different than solution - solution must be treated as global
            er_list.append(assemble(inner(velocity - sol, velocity - sol) * dx))  # faster
        global time_erc
        terc = toc() - tmp
        time_erc += terc
        print("Computed errornorm. Time: %f, Total: %f" % (terc, time_erc))

# ==Explicit Chorin method====================================================================================
if str_method == "chorinExpl":
    info("Initialization of explicit Chorin method")
    tic()

    # Boundary conditions
    bc0 = DirichletBC(V, noSlip, facet_function, 1)
    inflow = DirichletBC(V, v_in, facet_function, 2)
    outflow = DirichletBC(Q, 0.0, facet_function, 3)
    bcu = [inflow, bc0]
    bcp = [outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Create functions
    u0 = Function(V)
    if str_type == "pulsePrec":
        assign(u0, u_prec)
    if doSave:
        save_vel(u0, uFile)
        save_vel(u0, u2File)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity step
    F1 = (1 / k) * inner(u - u0, v) * dx + inner(grad(u0) * u0, v) * dx + \
         nu * inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q)) * dx
    L2 = -(1 / k) * div(u1) * q * dx

    # Velocity update
    a3 = inner(u, v) * dx
    L3 = inner(u1, v) * dx - k * inner(grad(p1), v) * dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
    info("Preconditioning: " + prec)

    # Time-stepping
    info("Running of explicit Chorin method")
    t = dt
    while t < (time + DOLFIN_EPS):
        print("t = ", t)

        # Update boundary condition
        v_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]  # PYTHON syntax
        try:
            solve(A1, u1.vector(), b1, "gmres", "default")
        except RuntimeError as inst:
            report_fail()
            exit()
        if doErrControl and round(t, 3) >= measure_time: compute_err(err_u2, u1, t)
        compute_div(div_u2, u1)
        if doSave:
            save_vel(u1, u2File)
            save_div(u1, d2File)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        try:
            solve(A2, p1.vector(), b2, "cg", prec)
        except RuntimeError as inst:
            report_fail()
            exit()
        if doSave:
            pFile << p1
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        try:
            solve(A3, u1.vector(), b3, "gmres", "default")
        except RuntimeError as inst:
            report_fail()
            exit()
        if doErrControl and round(t, 3) >= measure_time: compute_err(err_u, u1, t)
        compute_div(div_u, u1)
        if doSave:
            save_vel(u1, uFile)
            save_div(u1, dFile)
        end()

        # Move to next time step
        u0.assign(u1)
        t += dt

    info("Finished: explicit Chorin method")

# ==Direct method==============================================================================================
if str_method == "direct":
    info("Initialization of direct method")
    tic()

    # Define function spaces (Taylor-Hood)
    W = MixedFunctionSpace([V, Q])
    bc0 = DirichletBC(W.sub(0), noSlip, facet_function, 1)
    inflow = DirichletBC(W.sub(0), v_in, facet_function, 2)
    # Collect boundary conditions
    bcu = [inflow, bc0]

    # Define unknown and test function(s) NS
    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)

    # Define fields
    n = FacetNormal(mesh)
    I = Identity(u.geometric_dimension())  # Identity tensor
    x = SpatialCoordinate(mesh)
    theta = 0.5  # Crank-Nicholson

    # to assign solution in space W.sub(0) to Function(V) we need FunctionAssigner (cannot be assigned directly)
    fa = FunctionAssigner(V, W.sub(0))
    velSp = Function(V)

    # Define fields for time dependent case
    u0 = Function(V)  # velocity from previous time step
    if str_type == "pulsePrec":
        assign(u0, u_prec)
    if doSave:
        save_vel(u0, uFile)

    # Define steady part of the equation
    def T(u):
        return -p * I + 2.0 * nu * sym(grad(u))

    def F(u, v, q):
        return (inner(T(u), grad(v)) - q * div(u)) * dx + inner(grad(u) * u, v) * dx

    # Define variational forms
    F_ns = (inner((u - u0), v) / dt) * dx + (1.0 - theta) * F(u0, v, q) + theta * F(u, v, q)
    J_ns = derivative(F_ns, w)

    # NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns, form_compiler_parameters=ffc_options)
    NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns)
    # (var. formulation, unknown, Dir. BC, jacobian, optional)
    NS_solver = NonlinearVariationalSolver(NS_problem)

    prm = NS_solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-08
    prm['newton_solver']['relative_tolerance'] = 1E-08
    # prm['newton_solver']['maximum_iterations'] = 45
    # prm['newton_solver']['relaxation_parameter'] = 1.0
    prm['newton_solver']['linear_solver'] = 'mumps'  # or petsc,mumps   #ZKUSIT cyl3, jinak paralelizace

    # Time-stepping
    info("Running of direct method")
    t = dt
    while t < (time + DOLFIN_EPS):
        print("t = ", t)

        v_in.t = t

        # Compute
        begin("Solving NS ....")
        try:
            NS_solver.solve()
        except RuntimeError as inst:
            report_fail()
            exit()
        end()

        # Extract solutions:
        (u, p) = w.split()

        # we are assigning twice (now and inside save_vel), but it works with one method save_vel for direct and
        #   projection (we can split save_vel to save one assign)
        fa.assign(velSp, u)
        if doSave:
            save_vel(velSp, uFile)
            save_div(u, dFile)
            pFile << p
        compute_div(div_u, u)
        if doErrControl and round(t, 3) >= measure_time: compute_err(err_u, u, t)

        # Move to next time step
        assign(u0, u)
        t += dt

    info("Finished: direct method")

# ==Report====================================================================================
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
if doErrControl:
    total_err_u = math.sqrt(sum(err_u))
    total_err_u2 = math.sqrt(sum(err_u2))
    avg_err_u = total_err_u / math.sqrt(len(time_list))
    avg_err_u2 = total_err_u2 / math.sqrt(len(time_list))
    if time >= measure_time + 1 - DOLFIN_EPS:
        N = 1.0 / dt
        N0 = int(round(len(time_list) - N))
        N1 = int(round(len(time_list)))
        # last_cycle = time_list[N0:N1]
        # print("N: ",N," len: ",len(last_cycle), " list: ",last_cycle)
        last_cycle_err_u = math.sqrt(sum(err_u[N0:N1]) / N)
        last_cycle_div = sum(div_u[N0:N1]) / N
        last_cycle_err_min = math.sqrt(min(err_u[N0:N1]))
        last_cycle_err_max = math.sqrt(max(err_u[N0:N1]))
        if str_method == "chorinExpl":
            last_cycle_err_u2 = math.sqrt(sum(err_u2[N0:N1]) / N)
            last_cycle_div2 = sum(div_u2[N0:N1]) / N
            last_cycle_err_min2 = math.sqrt(min(err_u2[N0:N1]))
            last_cycle_err_max2 = math.sqrt(max(err_u2[N0:N1]))

    err_u = [math.sqrt(i) for i in err_u]
    err_u2 = [math.sqrt(i) for i in err_u2]

    # report of error norm for individual time steps
    with open(str_dir_name + "/report_err.csv", 'w') as reportFile:
        reportWriter = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
        reportWriter.writerow(["name"] + ["time"] + time_list)
        reportWriter.writerow([str_name] + ["corrected"] + err_u)
        if str_method == "chorinExpl":
            reportWriter.writerow([str_name] + ["tentative"] + err_u2)

# report of norm of div for individual time steps
with open(str_dir_name + "/report_div.csv", 'w') as reportFile:
    reportWriter = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportWriter.writerow([str_name] + ["corrected"] + div_u)
    if str_method == "chorinExpl":
        reportWriter.writerow([str_name] + ["tentative"] + div_u2)

# report without header
with open(str_dir_name + "/report.csv", 'w') as reportFile:
    reportWriter = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportWriter.writerow(
        ["pipe_test"] + [str_name] + [str_type] + [str_method] + [meshName] + [mesh] + [factor] + [time] + [dt] + [
            total - time_erc] + [time_erc] + [total_err_u] + [total_err_u2] + [avg_err_u] + [avg_err_u2] + [
            last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] + [last_cycle_div2] + [last_cycle_err_min] + [
            last_cycle_err_max] + [last_cycle_err_min2] + [last_cycle_err_max2])

# report with header
with open(str_dir_name + "/report_h.csv", 'w') as reportFile:
    reportWriter = csv.writer(reportFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportWriter.writerow(
        ["problem"] + ["name"] + ["type"] + ["method"] + ["mesh_name"] + ["mesh"] + ["factor"] + ["time"] + ["dt"] + [
            "timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"] + ["avg_err_u"] + [
            "avg_err_u2"] + ["last_cycle_err_u"] + ["last_cycle_err_u2"] + ["last_cycle_div"] + ["last_cycle_div2"] + [
            "last_cycle_err_min"] + ["last_cycle_err_max"] + ["last_cycle_err_min2"] + ["last_cycle_err_max2"])
    reportWriter.writerow(
        ["pipe_test"] + [str_name] + [str_type] + [str_method] + [meshName] + [mesh] + [factor] + [time] + [dt] + [
            total - time_erc] + [time_erc] + [total_err_u] + [total_err_u2] + [avg_err_u] + [avg_err_u2] + [
            last_cycle_err_u] + [last_cycle_err_u2] + [last_cycle_div] + [last_cycle_div2] + [last_cycle_err_min] + [
            last_cycle_err_max] + [last_cycle_err_min2] + [last_cycle_err_max2])

# create file showing all was done well
f = open(str_name + "_factor%4.2f_step_%dms_OK.report" % (factor, dt * 1000), "w")
f.close()
