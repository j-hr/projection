from __future__ import print_function
from dolfin import *
import sys
import argparse

import womersleyBC
import results
import problem as prb

# TODO authors, license


# Issues
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
# NT variant: use 1/3rd root of dim(V) instead of hmin

# Notes
# characteristic time for onset ~~ length of pipe/speed of fastest particle = 20(mm /factor*1081(mm/s) ~~  0.02 s/factor
# characteristic time for dt ~~ hmax/speed of fastest particle = hmax/(factor*1081(mm/s))
#   h = cubeRoot(volume/number of cells):
#   c1: 829 cells => h = 1.23 mm => 1.1 ms/factor
#   c2: 6632 cells => h = 0.62 mm => 0.57 ms/factor
#   c3: 53056 cells => h = 0.31 mm => 0.28 ms/factor

tc = results.TimeControl()
tc.init_watch('init', 'Initialization', True)
tc.init_watch('rhs', 'Assembled right hand side', True)
tc.init_watch('applybc1', 'Applied velocity BC 1st step', True)
tc.init_watch('applybc3', 'Applied velocity BC 3rd step', True)
tc.init_watch('applybcP', 'Applied pressure BC or othogonalized rhs', True)
tc.init_watch('assembleMatrices', 'Initial matrix assembly', False)
tc.init_watch('solve 1', 'Running solver on 1st step', True)
tc.init_watch('solve 2', 'Running solver on 2nd step', True)
tc.init_watch('solve 3', 'Running solver on 3rd step', True)
tc.init_watch('solve 4', 'Running solver on 4th step', True)
tc.init_watch('assembleA1', 'Assembled A1 matrix', True)
tc.start('init')
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
parameters["linear_algebra_backend"] = "PETSc"

# Debugging ============================================================================================================
# set_log_level(DEBUG)
PETScOptions.set('mat_mumps_icntl_4', 0)  # 1-3 gives lots of information for mumps direct solvers

# Resolve input arguments===============================================================================================
print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('method', help='Computing method', choices=['direct', 'chorinExpl', 'ipcs0', 'ipcs1'])
parser.add_argument('type', help='Flow type', choices=['steady', 'pulse0', 'pulsePrec'])
#   steady - parabolic profile (0.5 s onset)
# Womersley profile (1 s period)
#   pulse0 - u(0)=0
#   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
parser.add_argument('mesh', help='Mesh name')
parser.add_argument('time', help='Total time', type=int)
parser.add_argument('dt', help='Time step', type=float)
parser.add_argument('-F', '--factor', help='Velocity scale factor', type=float, default=1.0)
parser.add_argument('-e', '--error', help='Error control mode', choices=['doEC', 'noEC', 'test'], default='doEC')
parser.add_argument('-S', '--save', help='Save solution mode', choices=['doSave', 'noSave', 'diff'], default='noSave')
#   doSave: create .xdmf files with velocity, pressure, divergence
#   diff: save also difference vel-sol
#   noSave: do not create .xdmf files with velocity, pressure, divergence
parser.add_argument('-s', '--solvers', help='Solvers', choices=['direct', 'krylov'], default='krylov')
parser.add_argument('--prec', help='Preconditioner for pressure solver', choices=['hypre_amg', 'ilu'],
                    default='hypre_amg')
parser.add_argument('-p', '--precision', help='Krylov solver precision', type=int, default=6)
parser.add_argument('-b', '--bc', help='Pressure boundary condition mode', choices=['outflow', 'nullspace',
                                                                                    'nullspace_s', 'lagrange'],
                    default='outflow')
parser.add_argument('-n', '--name', default='test')
parser.add_argument('-r', help='Use rotation scheme', action='store_true')
parser.add_argument('-B', help='Use no BC in correction step', action='store_true')

args = parser.parse_args()
print(args)

problem = prb.Problem('pipe_womersley', args)

rm = results.ResultsManager(problem, tc)
print("Problem type: " + args.type)
print("Method:       " + args.method)
if args.method in ['chorinExpl', 'ipcs0', 'ipcs1']:
    problem.d()['hasTentativeVel'] = True
else:
    problem.d()['hasTentativeVel'] = False
if args.method == 'direct':
    if args.solver != 'default':
        exit('Parameter solvers should be \'default\' when using direct method.')
else:
    if args.solvers == 'krylov':
        print('Chosen Krylov solvers.')
    elif args.solvers == 'direct':
        print('Chosen direct solvers.')
useRotationScheme = False
if args.r:
    useRotationScheme = True

options = {'absolute_tolerance': 1e-25, 'relative_tolerance': 1e-12, 'monitor_convergence': True}

# Set parameter values
dt = args.dt
ttime = args.time
print("Time:         %1.0f s\ndt:           %d ms" % (ttime, 1000 * dt))
factor = args.factor
print("Velocity scale factor = %4.2f" % factor)
reynolds = 728.761 * factor
print("Computing with Re = %f" % reynolds)

if args.method == 'chorinExpl' and args.bc != 'outflow':
    exit('This combination is not coded properly.')

# ======================================================================================================================
# Import gmsh mesh
meshName = args.mesh
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")
dsIn = Measure("ds", subdomain_id=2, subdomain_data=facet_function)
dsOut = Measure("ds", subdomain_id=3, subdomain_data=facet_function)
dsWall = Measure("ds", subdomain_id=1, subdomain_data=facet_function)
rm.dsWall = dsWall
print("Mesh name: ", meshName, "    ", mesh)
print("Mesh norm max: ", mesh.hmax())
print("Mesh norm min: ", mesh.hmin())
problem.d()['mesh_info'] = str(mesh)
# ======================================================================================================================
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)  # pressure
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)
if args.bc == 'lagrange':
    L = FunctionSpace(mesh, "R", 0)
    QL = Q*L

# fixed parameters (used in analytic solution and in BC)
nu = 3.71  # kinematic viscosity
R = 5.0  # cylinder radius

# in/outflow area
area = assemble(interpolate(Expression("1.0"), Q) * dsIn)

volume = assemble(interpolate(Expression("1.0"), Q) * dx)

# Boundary Conditions===================================================================================================
# boundary parts: 1 walls, 2 inflow, 3 outflow
noSlip = Constant((0.0, 0.0, 0.0))
if args.type == "steady":
    v_in = Expression(("0.0", "0.0",
                       "(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):\
                       (factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),
                      t=0, factor=factor)
else:
    # update by assigning precomputed analytic solution
    v_in = Function(V)

# Initial Conditions====================================================================================================
if args.type == "pulsePrec":
    # computes initial velocity as a solution of steady Stokes problem with input velocity v_in
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
        # lhs(F) == rhs(F) means that is a linear problem
        solve(lhs(F) == rhs(F), w, bcu, solver_parameters={'linear_solver': 'mumps'})
    except RuntimeError as inst:
        rm.report_fail(args.name, dt, t)
        exit()
    # Extract solutions:
    (u_prec, p_prec) = w.split()
    print("Computed initial velocity. Time:%f" % (toc() - temp))
    end()

    # plot(u_prec, mode = "glyphs", title="steady solution", interactive=True)
    # exit()

# Output and error control =============================================================================================
rm.initialize(V, Q, mesh, "%sresults_%s_%s_%s_%s_factor%4.2f_%ds_%dms" %
              (args.name, args.type, args.method, args.bc, meshName, factor, ttime, dt * 1000), factor, PS, V, meshName, dt)


def averaging_pressure(pressure):
    # averaging pressure (substract average)
    p_average = assemble((1.0/volume) * pressure * dx)
    # print('Average pressure: ', p_average)
    p_average_function = interpolate(Expression("p", p=p_average), Q)
    # print(p_average_function, pressure, pressure_Q)
    pressure.assign(pressure - p_average_function)


def save_pressure(is_tent, pressure):
    # Report pressure gradient
    p_in = assemble((1.0/area) * pressure * dsIn)
    p_out = assemble((1.0/area) * pressure * dsOut)
    # print(p_in, p_(0.0, 0.0, -10), p_(0.0, 5.0, -10.0), p_(5.0, 0.0, -10.0))
    p_diff = (p_out - p_in)/20.0  # 20.0 is a length of a pipe
    # plot(pressure, interactive=True)
    # exit()
    rm.save_pressure(is_tent, pressure, p_diff)


# Common functions =====================================================================================================
solver_vel = None   # solver for 1st and 3rd step (tentative and corrected velocity)
solver_p = None     # solver for 2nd step (pressure or tentative pressure for rotation scheme)
solver_rot = None   # solver for 4th step (final pressure for rotation scheme)
# solver_p and solver_rot must be different, because of preserving structure of matrix for preconditioning
# 2nd step is Poisson problem, 4th step is just a mass matrix
null_space = None


def set_projection_solvers():
    global solver_p, solver_vel, solver_rot, null_space

    if args.solvers == 'direct':
        solver_vel = LUSolver('umfpack')
        solver_p = LUSolver('umfpack')
        if useRotationScheme:
            solver_rot = LUSolver('umfpack')
    else:
        solver_vel = KrylovSolver('gmres', 'ilu')   # nonsymetric > gmres  # IFNEED try hypre_amg
        solver_p = KrylovSolver('cg', args.prec)          # symmetric > CG
        # solver_p = KrylovSolver('gmres', 'hypre_amg')     # NT this, with disabled setnullspace gives same oscilations
        if useRotationScheme:
            solver_rot = KrylovSolver('cg', args.prec)

        options = {'absolute_tolerance': 10E-12, 'relative_tolerance': 10**(-args.precision),
                   'monitor_convergence': True, 'maximum_iterations': 500}

    # Get the nullspace if there are no pressure boundary conditions
    foo = Function(Q)     # auxiliary vector for setting pressure nullspace
    if args.bc in ['nullspace', 'nullspace_s']:
        null_vec = Vector(foo.vector())
        Q.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm('l2')
        null_space = VectorSpaceBasis([null_vec])
        if args.bc == 'nullspace':
            as_backend_type(A2).set_nullspace(null_space)
        if args.bc == 'nullspace_s':
            solver_p.set_nullspace(null_space)
            # IMP deprecated for KrylovSolver, not working for direct solver
            # not working with PETSc > 3.5

    # apply global options for Krylov solvers
    if args.solvers == 'krylov':
        for solver in [solver_vel, solver_p, solver_rot] if useRotationScheme else [solver_vel, solver_p]:
            for key, value in options.items():
                try:
                    solver.parameters[key] = value
                except KeyError:
                    print('Invalid option %s for KrylovSolver' % key)
                    exit()
            solver.parameters['preconditioner']['structure'] = 'same'


# Explicit Chorin method================================================================================================
if args.method == "chorinExpl":
    info("Initialization of explicit Chorin method")

    # Boundary conditions
    bc0 = DirichletBC(V, noSlip, facet_function, 1)
    inflow = DirichletBC(V, v_in, facet_function, 2)
    outflow = DirichletBC(Q, 0.0, facet_function, 3)
    bcu = [inflow, bc0]
    bcp = [outflow]

    # Define and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Create functions
    u0 = Function(V)
    if args.type == "pulsePrec":
        assign(u0, u_prec)
    if rm.doSave:
        rm.save_vel(False, u0, 0.0)
        rm.save_vel(True, u0, 0.0)
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

    set_projection_solvers()

    # Time-stepping
    info("Running of explicit Chorin method")
    t = dt
    while t < (ttime + dt/2.0):
        print("t = ", t)
        rm.update_time(t)

        # Update boundary condition
        v_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        try:
            solver_vel.solve(A1, u1.vector(), b1)
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        rm.compute_err(True, u1, t)
        rm.compute_div(True, u1)
        if rm.doSave:
            rm.save_vel(True, u1, t)
            rm.save_div(True, u1)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        try:
            solver_p.solve(A2, p1.vector(), b2)
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        averaging_pressure(p1)
        save_pressure(False, p1)
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        try:
            solve(A3, u1.vector(), b3, "gmres", "default")
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        rm.compute_err(False, u1, t)
        rm.compute_div(False, u1)
        if rm.doSave:
            rm.save_vel(False, u1, t)
            rm.save_div(False, u1)
        end()

        # Move to next time step
        u0.assign(u1)
        t += dt

    info("Finished: explicit Chorin method")

# Incremental pressure correction scheme with explicit nonlinear term ==================================================
# incremental = extrapolate pressure from previous steps (here: use previous step)
# viscosity term treated semi-explicitly (Crank-Nicholson)
if args.method == 'ipcs0':
    info("Initialization of Incremental pressure correction scheme n. 0")

    # Boundary conditions
    bc0 = DirichletBC(V, noSlip, facet_function, 1)
    inflow = DirichletBC(V, v_in, facet_function, 2)
    outflow = DirichletBC(Q, 0.0, facet_function, 3)  # we can choose, whether to use it, or use projection to nullspace
    bcu = [inflow, bc0]
    bcp = []                    # QQ can I use pressure BC when I use grad(p) in velocity?
    if args.bc == 'outflow':             # QQ what approach is better? Compare!
        bcp = [outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Initial conditions
    u0 = Function(V)
    p0 = Function(Q)
    if args.type == "pulsePrec":
        assign(u0, u_prec)
        assign(p0, p_prec)
    if rm.doSave:
        rm.save_vel(False, u0, 0.0)
        rm.save_vel(True, u0, 0.0)

    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity, solve to u1
    U = 0.5*(u + u0)
    F1 = (1./k)*inner(u - u0, v)*dx + inner(dot(grad(u0), u0), v)*dx\
        + nu*inner(grad(U), grad(v))*dx + inner(grad(p0), v)*dx\
        - inner(f, v)*dx
    a1, L1 = system(F1)

    # Projection, solve to p1
    F2 = inner(grad(p - p0), grad(q))*dx + (1./k)*q*div(u1)*dx
    a2, L2 = system(F2)

    # Finalize, solve to u1
    F3 = (1./k)*inner(u - u1, v)*dx + inner(grad(p1 - p0), v)*dx
    a3, L3 = system(F3)

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    set_projection_solvers()

    # Time-stepping
    info("Running of Incremental pressure correction scheme n. 0")
    t = dt
    while t < (ttime + dt/2.0):
        print("t = ", t)
        rm.update_time(t)

        # Update boundary condition
        v_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b = assemble(L1)
        [bc.apply(A1, b) for bc in bcu]
        try:
            solver_vel.solve(A1, u1.vector(), b)
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        rm.compute_err(True, u1, t)
        rm.compute_div(True, u1)
        if rm.doSave:
            rm.save_vel(True, u1, t)
            rm.save_div(True, u1)
        end()

        begin("Computing pressure correction")
        b = assemble(L2)
        [bc.apply(A2, b) for bc in bcp]
        if args.bc == 'nullspace':
            null_space.orthogonalize(b)
        try:
            solver_p.solve(A2, p1.vector(), b)
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        averaging_pressure(p1)
        save_pressure(False, p1)
        end()

        b = assemble(L3)
        [bc.apply(A3, b) for bc in bcu]
        try:
            solver_vel.solve(A3, u1.vector(), b)
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        rm.compute_err(False, u1, t)
        rm.compute_div(False, u1)
        if rm.doSave:
            rm.save_vel(False, u1, t)
            rm.save_div(False, u1)
        end()

        # Move to next time step
        p0.assign(p1)
        u0.assign(u1)
        t += dt

    info("Finished: Incremental pressure correction scheme n. 0")

# Incremental pressure correction with nonlinearity treated by Adam-Bashword + Crank-Nicolson. =========================
# incremental = extrapolate pressure from previous steps (here: use previous step)
# viscosity term treated semi-explicitly (Crank-Nicholson)
if args.method == 'ipcs1':
    info("Initialization of Incremental pressure correction scheme n. 1")
    n = FacetNormal(mesh)

    # Boundary conditions
    bc0 = DirichletBC(V, noSlip, facet_function, 1)
    inflow = DirichletBC(V, v_in, facet_function, 2)
    # we can choose, whether to use pressure BC, or use projection to nullspace
    outflow = DirichletBC(Q, Constant(0.0), facet_function, 3)
    # outflow = DirichletBC(Q, Constant(0.0), "near(x[0],0.0) && near(x[1],0.0) && near(x[2],10.0)", method="pointwise")
    bcu = [inflow, bc0]
    bcp = []
    if args.bc == 'outflow':
        bcp = [outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    if args.bc == 'lagrange':
        (pQL, rQL) = TrialFunction(QL)
        (qQL, lQL) = TestFunction(QL)
    else:
        p = TrialFunction(Q)
        q = TestFunction(Q)

    # Initial conditions
    u0 = Function(V)  # velocity at previous time step
    u1 = Function(V)  # velocity two time steps back
    p0 = Function(Q)  # previous pressure

    if args.type == "pulsePrec":
        assign(u0, u_prec)
        assign(u1, u_prec)
        assign(p0, p_prec)
    if rm.doSave:
        rm.save_vel(False, u0, 0.0)
        rm.save_vel(True, u0, 0.0)

    u_ = Function(V)         # current tentative velocity
    u_cor = Function(V)         # current corrected velocity
    if args.bc == 'lagrange':
        p_QL = Function(QL)    # current pressure or pressure help function from rotation scheme
        pQ = Function(Q)     # auxiliary function for conversion between QL.sub(0) and Q
    else:
        p_ = Function(Q)         # current pressure or pressure help function from rotation scheme
    p_mod = Function(Q)      # current modified pressure from rotation scheme

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity, solve to u_
    U = 0.5*(u + u0)
    U_ = 1.5*u0 - 0.5*u1

    nonlinearity = inner(dot(grad(U), U_), v)*dx

    F1 = (1./k)*inner(u - u0, v)*dx + nonlinearity\
        + nu*inner(grad(U), grad(v))*dx + inner(grad(p0), v)*dx\
        - inner(f, v)*dx     # solve to u_
    a1, L1 = system(F1)

    if useRotationScheme:
        # Rotation scheme
        if args.bc == 'lagrange':
            F2 = inner(grad(pQL), grad(qQL))*dx + (1./k)*qQL*div(u_)*dx + pQL*lQL*dx + qQL*rQL*dx
        else:
            F2 = inner(grad(p), grad(q))*dx + (1./k)*q*div(u_)*dx
    else:
        # Projection, solve to p_
        if args.bc == 'lagrange':
            F2 = inner(grad(pQL - p0), grad(qQL))*dx + (1./k)*qQL*div(u_)*dx + pQL*lQL*dx + qQL*rQL*dx
        else:
            F2 = inner(grad(p - p0), grad(q))*dx + (1./k)*q*div(u_)*dx
    a2, L2 = system(F2)

    # Finalize, solve to u_
    if useRotationScheme:
        # Rotation scheme
        if args.bc == 'lagrange':
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_QL.sub(0)), v)*dx
        else:
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_), v)*dx
    else:
        if args.bc == 'lagrange':
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_QL.sub(0) - p0), v)*dx
        else:
            F3 = (1./k)*inner(u - u_, v)*dx + inner(grad(p_ - p0), v)*dx
    a3, L3 = system(F3)

    if useRotationScheme:
        # Rotation scheme: modify pressure
        if args.bc == 'lagrange':
            pr = TrialFunction(Q)
            qr = TestFunction(Q)
            F4 = (pr - p0 - p_QL.sub(0) + nu*div(u_))*qr*dx
        else:
            F4 = (p - p0 - p_ + nu*div(u_))*q*dx
        # TODO zkusit, jestli to nebude rychlejsi? nepocitat soustavu, ale p.assign(...), nutno project(div(u),Q) coz je pocitani podobne soustavy
        # TODO zkusit v project zadat solver_type='lu' >> primy resic by mel byt efektivnejsi
        a4, L4 = system(F4)

    # Assemble matrices
    tc.start('assembleMatrices')
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    if useRotationScheme:
        A4 = assemble(a4)
    tc.end('assembleMatrices')

    set_projection_solvers()

    if args.bc == 'lagrange':
        fa = FunctionAssigner(Q, QL.sub(0))

    # Time-stepping
    tc.end('init')
    info("Running of Incremental pressure correction scheme n. 1")
    t = dt
    while t < (ttime + dt/2.0):
        print("t = ", t)
        rm.update_time(t)

        # Update boundary condition
        v_in.assign(rm.solution)

        # assemble matrix (ir depends on solution)
        tc.start('assembleA1')
        assemble(a1, tensor=A1)
        tc.end('assembleA1')

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        tc.start('rhs')
        b = assemble(L1)
        tc.end('rhs')
        tc.start('applybc1')
        [bc.apply(A1, b) for bc in bcu]
        tc.end('applybc1')
        try:
            tc.start('solve 1')
            solver_vel.solve(A1, u_.vector(), b)
            tc.end('solve 1')
        except RuntimeError as inst:
            rm.report_fail(t)
            exit()
        rm.compute_err(True, u_, t)
        rm.compute_div(True, u_)
        if rm.doSave:
            rm.save_vel(True, u_, t)
            rm.save_div(True, u_)
        end()

        if useRotationScheme:
            begin("Computing tentative pressure")
        else:
            begin("Computing pressure")
        tc.start('rhs')
        b = assemble(L2)
        tc.end('rhs')
        tc.start('applybcP')
        [bc.apply(A2, b) for bc in bcp]
        if args.bc in ['nullspace', 'nullspace_s']:
            null_space.orthogonalize(b)
        tc.end('applybcP')
        # print(A2, b, p_.vector())
        try:
            tc.start('solve 2')
            if args.bc == 'lagrange':
                solver_p.solve(A2, p_QL.vector(), b)
            else:
                solver_p.solve(A2, p_.vector(), b)
            tc.end('solve 2')
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        if useRotationScheme:
            foo = Function(Q)
            if args.bc == 'lagrange':
                fa.assign(pQ, p_QL.sub(0))
                foo.assign(pQ + p0)
            else:
                foo.assign(p_+p0)
            averaging_pressure(foo)
            save_pressure(True, foo)
        else:
            if args.bc == 'lagrange':
                fa.assign(pQ, p_QL.sub(0))
                averaging_pressure(pQ)
                save_pressure(False, pQ)
            else:
                averaging_pressure(p_)
                save_pressure(False, p_)

        end()

        begin("Computing corrected velocity")
        tc.start('rhs')
        b = assemble(L3)
        tc.end('rhs')
        if not args.B:
            tc.start('applybc3')
            [bc.apply(A3, b) for bc in bcu]
            tc.end('applybc3')
        try:
            tc.start('solve 3')
            solver_vel.solve(A3, u_cor.vector(), b)
            tc.end('solve 3')
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        rm.compute_err(False, u_cor, t)
        rm.compute_div(False, u_cor)
        if rm.doSave:
            rm.save_vel(False, u_cor, t)
            rm.save_div(False, u_cor)
        end()

        if useRotationScheme:
            begin("Rotation scheme pressure correction")
            tc.start('rhs')
            b = assemble(L4)
            tc.end('rhs')
            try:
                tc.start('solve 4')
                solver_rot.solve(A4, p_mod.vector(), b)
                tc.end('solve 4')
            except RuntimeError as inst:
                rm.report_fail(args.name, dt, t)
                exit()
            averaging_pressure(p_mod)
            save_pressure(False, p_mod)
            end()

        # plot(p_, title='tent')
        # plot(p_mod, title='cor', interactive=True)
        # exit()

        # compute force on wall
        rm.compute_force(u_cor, p_mod if useRotationScheme else (pQ if args.bc == 'lagrange' else p_), n, t)  # NT clean somehow?

        # Move to next time step
        u1.assign(u0)
        u0.assign(u_cor)
        if useRotationScheme:
            p0.assign(p_mod)
        else:
            if args.bc == 'lagrange':
                p0.assign(pQ)
            else:
                p0.assign(p_)

        t += dt

    info("Finished: Incremental pressure correction scheme n. 1")

# Direct method=========================================================================================================
if args.method == "direct":
    info("Initialization of direct method")

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
    if args.type == "pulsePrec":
        assign(u0, u_prec)
    if rm.doSave:
        rm.save_vel(False, u0, 0.0)

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
    prm['newton_solver']['linear_solver'] = 'mumps'  # TODO try 'superlu' (for killed)

    # Time-stepping
    info("Running of direct method")
    t = dt
    while t < (ttime + dt/2.0):
        print("t = ", t)
        rm.update_time(t)

        v_in.t = t

        # Compute
        begin("Solving NS ....")
        try:
            tc.start('solve 1')
            NS_solver.solve()
            tc.end('solve 1')
        except RuntimeError as inst:
            rm.report_fail(args.name, dt, t)
            exit()
        end()

        # Extract solutions:
        (u, p) = w.split()

        # we are assigning twice (now and inside save_vel), but it works with one method save_vel for direct and
        #   projection (we can split save_vel to save one assign)
        fa.assign(velSp, u)
        if rm.doSave:
            rm.save_vel(False, velSp, t)
            rm.save_div(False, u)
        averaging_pressure(p1)
        save_pressure(False, p1)
        rm.compute_div(False, u)
        rm.compute_err(False, u, t)

        # Move to next time step
        assign(u0, u)
        t += dt

    info("Finished: direct method")

# Report
rm.report()
