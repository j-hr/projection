from __future__ import print_function
from dolfin import *
import math #TODO add "math."
import csv
import sys

#TODO if method = ... >> if hasTentative

#check number of arguments
nargs = 4
arguments = "method type T dt"
if len(sys.argv) <> nargs+1:
    exit("Wrong number of arguments. Should be: %d (%s)"%(nargs,arguments))

#choose a method: direct, chorinExpl
str_method=sys.argv[1]

#choose type of flow:
#   steady - parabolic profile (0.5 s onset)
#Womersley profile (1 s period)
#   pulse0 - u(0)=0
#   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
str_type=sys.argv[2]

# TODO only when needed:
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
#ffc_options = {"quadrature_degree": 5}
#parameters["allow_extrapolation"] = True 

# Import gmsh mesh 
meshname = "cyl_c1"
mesh = Mesh("meshes/"+meshname+".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_facet_region.xml")

# Set parameter values
dt = float(sys.argv[4])
Time = float(sys.argv[3])

# fixed parameters (used in analytic solution)
nu = 3.71 #kinematic viscosity
R=5.0 # cylinder radius

#==Boundary Conditions================================================================================
# boundary parts: 1 walls, 2 inflow, 3 outflow
noslip  = Constant((0.0, 0.0, 0.0))
# Womersley profile (1 s period)
#TODO pulse0 - u(0)=0
#TODO pulsePrec - u(0) from precomputed solution (steady Stokes problem)
if str_type == "steady" :
    v_in = Expression(("0.0","0.0","(t<0.5)?((sin(pi*t))*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"),t=0)
elif str_type == "pulse0" :
    v_in = Expression(("0.0","0.0","(t<0.5)?((sin(pi*t))*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"),t=0)
elif str_type == "pulsePrec" :
    v_in = Expression(("0.0","0.0","(t<0.5)?((sin(pi*t))*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"),t=0)

#==Output settings====================================================================================
# Create files for storing solution
str_name = "pipe_test_"+str_type+"_"+str_method+"_"+meshname+"_%ds_%dms"%(Time,dt*1000)
ufile = File("results_"+str_name+"/velocity.xdmf")
dfile = File("results_"+str_name+"/divergence.xdmf") # maybe just compute norm
pfile = File("results_"+str_name+"/pressure.xdmf")
if str_method=="chorinExpl" : 
    u2file = File("results_"+str_name+"/velocity_tent.xdmf")
    d2file = File("results_"+str_name+"/div_tent.xdmf") # maybe just compute norm

#==Analytic solution====================================================================================
if str_type == "steady" :
    solution = Expression(("0.0","0.0","(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"))
elif (str_type == "pulse0") or (str_type == "pulsePrec") :
    solution = Expression(("0.0","0.0","(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))")) #TODO


#==Error control====================================================================================
# TODO aternative error control? 
measure_time = 0.5 if str_type=="steady" else 1 # maybe change
timelist=[]
er_u = []
er_u2 = []
time_erc = 0


def computeErr(erlist,velocity):
    if timelist.last > t : timelist.append(t)
    terc=toc()
    erlist.append(errornorm(velocity, solution, norm_type='l2', degree_rise=3))
    time_erc += toc() - terc


#==Explicite Chorin method====================================================================================
if str_method=="chorinExpl" :
    info("Initialization of explicite Chorin method")
    tic()

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    # Boundary conditions
    bc0     = DirichletBC(V, noslip, facet_function,1)
    inflow	= DirichletBC(V, v_in, facet_function,2)
    outflow = DirichletBC(Q, 0.0, facet_function,3)
    bcu = [inflow,bc0]
    bcp = [outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
        nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = -(1/k)*div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
    info("Preconditioning: " + prec)

    # Create files for storing solution
    out = output()

    # Time-stepping
    info("Running of explicite Chorin method")
    t = dt
    while t < Time + DOLFIN_EPS:
        print("t = ", t)
        
        # Update boundary condition
        v_in.t = t
        
        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "gmres", "default")
        u2file << u1
        if t>measure_time : computeErr(er_u2,u1)
        #save DIV
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        solve(A2, p1.vector(), b2, "cg", prec)
        pfile << p1
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "gmres", "default")
        ufile << u1
        if t>measure_time : computeErr(er_u,u1)
        #save DIV
        end()

        # Move to next time step
        u0.assign(u1)
        t += dt

    info("Finished: explicite Chorin method")
    total=toc()

#==Direct method==============================================================================================
if str_method=="direct" :
    info("Initialization of direct method")
    tic()

    # Define function spaces (Taylor-Hood)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = MixedFunctionSpace([V,Q])
    bc0     = DirichletBC(W.sub(0), noslip, facet_function,1)
    inflow	= DirichletBC(W.sub(0), v_in, facet_function,2)
    # Collect boundary conditions
    bcu = [inflow,bc0]

    # Define unknown and test function(s) NS
    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)

    # Define fields
    n = FacetNormal(mesh)
    I = Identity(u.geometric_dimension())    # Identity tensor
    x = SpatialCoordinate(mesh)
    theta = 0.5 #Crank-Nicholson

    #Define fields for time dependent case
    u0 = Function(V) #velocity from previous time step

    # Define steady part of the equation
    def T(u):
        return -p*I + 2.0*nu*sym(grad(u))

    def F(u, v, q):
        return (inner(T(u), grad(v)) - q*div(u))*dx + inner(grad(u)*u, v)*dx 

    # Define variational forms
    F_ns = (inner((u-u0),v)/dt)*dx + (1.0-theta)*F(u0,v,q) + theta*F(u,v,q)
    J_ns=derivative(F_ns,w)

    #NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns, form_compiler_parameters=ffc_options)
    NS_problem = NonlinearVariationalProblem(F_ns, w, bcu, J_ns)
    # (var.formulace, neznama, Dir.okrajove podminky, jakobian, optional)
    NS_solver  = NonlinearVariationalSolver(NS_problem)

    #	prm = NS_solver.parameters
    #prm['newton_solver']['absolute_tolerance'] = 1E-08
    #prm['newton_solver']['relative_tolerance'] = 1E-08
    #prm['newton_solver']['maximum_iterations'] = 45
    #prm['newton_solver']['relaxation_parameter'] = 1.0
    #prm['newton_solver']['linear_solver'] = 'mumps' #or petsc,mumps

    # Create files for storing solution
    out = output()

    # Time-stepping
    info("Running of direct method")
    t = dt
    while t < Time + DOLFIN_EPS:
        print("t = ", t)
        
        v_in.v=t
        
        # Compute
        begin("Solving NS ....")
        NS_solver.solve()
        end()

        # Extract solutions:
        (u, p) = w.split()

        ufile << u
        #save DIV
        pfile << p
        if t>measure_time : computeErr(er_u,u)

        # Move to next time step
        assign(u0,u)
        t += dt

    info("Finished: direct method")
    total=toc()

#==Report====================================================================================
total_err_u = math.sqrt(sum(err_u))
total_err_u2 = math.sqrt(sum(err_u2))
with open("results_"+str_name+"/report_err.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow(timelist)
    reportwriter.writerow(err_u)
    if str_method=="chorinExpl" : reportwriter.writerow(err_u2)

with open("results_"+str_name+"/report.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow(["pipe_test"]+[str_type]+[str_method]+[mesh]+[Time]+[dt]+[total-time_erc]+[time_erc])

with open("results_"+str_name+"/report_h.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow(["problem"] + ["type"] + ["method"] + ["meshname"]+ ["time"] + ["dt"] + ["timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"])
    reportwriter.writerow(["pipe_test"]+[str_type]+[str_method]+[mesh]+[Time]+[dt]+[total-time_erc]+[time_erc]+[total_err_u]+[total_err_u2])

