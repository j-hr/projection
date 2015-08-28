from __future__ import print_function
from dolfin import *
import sys

import womersleyBC
import results


# Reduce HDD usage
# IFNEED option to set which steps to save (for example save only (t mod 0.05) == 0 )

# Reorganize code
# TODO clean results.py
# FUTURE (adapting for different geometry) move IC, BC, ?mesh to problem file

# Continue work
# TODO another projection methods (MiroK)
# TODO ipcs0
# TODO ipcs1
# TODO rotational scheme
# TODO if method == ... change to if hasTentative

# Issues
# QQ How to be sure that analytic solution is right?
# TODO check compatibility of solution and inflow condition it should be equal
# TODO check if errornorm and integral give the same results
# TODO cyl3 cannot be solved directly, pulsePrec also fails. try mumps, then paralelize
# mesh.hmax() returns strange values >> hmin() is better
#   Mesh name:  cyl_c1      <Mesh of topological dimension 3 (tetrahedra) with 280 vertices and 829 cells, ordered>
#   Mesh norm max:  6.23747141372
#   Mesh norm min:  2.23925109892
#   Mesh name:  cyl_c2      <Mesh of topological dimension 3 (tetrahedra) with 1630 vertices and 6632 cells, ordered>
#   Mesh norm max:  13.5627060087
#   Mesh norm min:  1.06943898166
#   Mesh name:  cyl_c3      <Mesh of topological dimension 3 (tetrahedra) with 10859 vertices and 53056 cells, ordered>
#   Mesh norm max:  11.1405707495
#   Mesh norm min:  0.525851168761

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

# choose type of flow:
#   steady - parabolic profile (0.5 s onset)
# Womersley profile (1 s period)
#   pulse0 - u(0)=0
#   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
str_type = sys.argv[2]
print("Problem type: " + str_type)

# save mode
#   save: create .xdmf files with velocity, pressure, divergence
#   noSave: do not create .xdmf files with velocity, pressure, divergence
rm = results.ResultsManager()
rm.set_save_mode(sys.argv[8])

# error control mode
#   0 no error control
#   -1 default option (start measuring error at 0.5 for steady and 1.0 for unsteady flow)
rm.set_error_control_mode(sys.argv[7], str_type)

# choose a method: direct, chorinExpl
str_method = sys.argv[1]
print("Method:       " + str_method)
hasTentativeVel = False
if str_method == 'chorinExpl':
    hasTentativeVel = True
    rm.hasTentativeVel = True

# Set parameter values
dt = float(sys.argv[5])
ttime = float(sys.argv[4])
print("Time:         %1.0f s\ndt:           %d ms" % (ttime, 1000 * dt))
factor = float(sys.argv[6])  # default: 1.0
print("Velocity scale factor = %4.2f" % factor)
reynolds = 728.761 * factor
print("Computing with Re = %f" % reynolds)

# Import gmsh mesh 
meshName = sys.argv[3]
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")
print("Mesh name: ", meshName, "    ", mesh)
print("Mesh norm max: ", mesh.hmax())
print("Mesh norm min: ", mesh.hmin())
# ======================================================================================================================
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

# fixed parameters (used in analytic solution and in BC)
nu = 3.71  # kinematic viscosity
R = 5.0  # cylinder radius

# MOVE#==Boundary Conditions================================================================================
# boundary parts: 1 walls, 2 inflow, 3 outflow
noSlip = Constant((0.0, 0.0, 0.0))
if str_type == "steady":
    v_in = Expression(("0.0", "0.0",
                       "(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):\
                       (factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),
                      t=0, factor=factor)
elif str_type == "pulse0" or str_type == "pulsePrec":
    womersleyBC.factor = factor
    v_in = womersleyBC.WomersleyProfile()

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
    try:
        solve(lhs(F) == rhs(F), w, bcu)  # QQ why this way?
    except RuntimeError as inst:
        rm.report_fail(str_name, factor, dt, t)
        exit()
    # Extract solutions:
    (u_prec, p_prec) = w.split()
    print("Computed initial velocity. Time:%f" % (toc() - temp))
    end()

    # plot(u_prec, mode = "glyphs", title="steady solution", interactive=True)
    # exit()

# Output and error control =============================================================================================
rm.initialize_output(V, mesh, "%sresults_%s_%s_%s_factor%4.2f_%ds_%dms" % (str_name, str_type, str_method, meshName,
                                                                           factor, ttime, dt * 1000))
rm.initialize_error_control(str_type, factor, PS, V, meshName)

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
    if rm.doSave:
        rm.save_vel(False, u0)
        rm.save_vel(True, u0)
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
    while t < (ttime + DOLFIN_EPS):
        print("t = ", t)
        do_EC = rm.do_compute_error(t)

        # Update boundary condition
        v_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]  # PYTHON syntax
        try:
            solve(A1, u1.vector(), b1, "gmres", "default")
        except RuntimeError as inst:
            rm.report_fail(str_name, factor, dt, t)
            exit()
        if do_EC:
            rm.compute_err(True, u1, t, str_type, V, factor)
        rm.compute_div(True, u1)
        if rm.doSave:
            rm.save_vel(True, u1)
            rm.save_div(True, u1)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        try:
            solve(A2, p1.vector(), b2, "cg", prec)
        except RuntimeError as inst:
            rm.report_fail(str_name, factor, dt, t)
            exit()
        if rm.doSave:
            rm.pFile << p1
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        try:
            solve(A3, u1.vector(), b3, "gmres", "default")
        except RuntimeError as inst:
            rm.report_fail(str_name, factor, dt, t)
            exit()
        if do_EC:
            rm.compute_err(False, u1, t, str_type, V, factor)
        rm.compute_div(False, u1)
        if rm.doSave:
            rm.save_vel(False, u1)
            rm.save_div(False, u1)
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
    if rm.doSave:
        rm.save_vel(False, u0)

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
    while t < (ttime + DOLFIN_EPS):
        print("t = ", t)
        do_EC = rm.do_compute_error(t)

        v_in.t = t

        # Compute
        begin("Solving NS ....")
        try:
            NS_solver.solve()
        except RuntimeError as inst:
            rm.report_fail(str_name, factor, dt, t)
            exit()
        end()

        # Extract solutions:
        (u, p) = w.split()

        # we are assigning twice (now and inside save_vel), but it works with one method save_vel for direct and
        #   projection (we can split save_vel to save one assign)
        fa.assign(velSp, u)
        if rm.doSave:
            rm.save_vel(False, velSp)
            rm.save_div(False, u)
            rm.pFile << p
        rm.compute_div(False, u)
        if do_EC:
            rm.compute_err(False, u, t, str_type, V, factor)

        # Move to next time step
        assign(u0, u)
        t += dt

    info("Finished: direct method")

# Report
rm.report(dt, ttime, str_name, str_type, str_method, meshName, mesh, factor)
