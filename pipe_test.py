from __future__ import print_function
from dolfin import *
from math import *
import csv
import sys

#check number of arguments
nargs = 3
arguments = "T dt nu"
if len(sys.argv) <> nargs+1:
  exit("Wrong number of arguments. Should be: %d (%s)"%(nargs,arguments))


# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
#ffc_options = {"quadrature_degree": 5}
#parameters["allow_extrapolation"] = True 

# Import gmsh mesh 
meshname = "cylinder24k"
mesh = Mesh("meshes/"+meshname+".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_facet_region.xml")

# Set parameter values
dt = float(sys.argv[2])
Time = float(sys.argv[1])
nu = float(sys.argv[3]) #nu = 0.01
r0=5.0 # polomer valce
c0=0.25 # faktor rychlosti
freq=1 # pulse frequency modificator

#oznaceni ulohy
str_problem="pulse"
str_meshtype="gmsh"
str_note="comparison"    
str_method="to be initialized"
#str_name="to be set"

#run problems
info("Initialization of explicite Chorin method")
str_method="explChor"
tic()

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
#W = MixedFunctionSpace([V, Q])

# oznaceni casti hranice: 1 walls, 2 inflow, 3 outflow
noslip  = Constant((0.0, 0.0, 0.0))
#v_in = Expression(("0.0","0.0","c*v*(r*r-(x[0]*x[0]+x[1]*x[1]))"),r=r0,v=0,c=0.25)
v_in = Expression(("0.0","0.0","c*(0.5+0.5*sin(pi*v*fr))*(r*r-(x[0]*x[0]+x[1]*x[1]))"),r=r0,v=0,c=c0,fr=freq)
bc0     = DirichletBC(V, noslip, facet_function,1)
inflow	= DirichletBC(V, v_in, facet_function,2)
outflow = DirichletBC(Q, 0.0, facet_function,3)
#bc0     = DirichletBC(W.sub(0), noslip, facet_function,1)
#inflow	= DirichletBC(W.sub(0), v_in, facet_function,2)
#outflow = DirichletBC(W.sub(1), 0.0, facet_function,3)
bcu = [inflow,bc0]
bcp = [outflow]

# Create files for storing solution
def output():
  global str_name
  str_name=str_problem+"_%4.2f_"%(nu)+str_method+"_"+str_meshtype+"_"+meshname+"_"+str_note     
  ufile = File("results_%s_%ds_%dms/velocity.xdmf"%(str_name,Time,dt*1000))
  pfile = File("results_%s_%ds_%dms/pressure.xdmf"%(str_name,Time,dt*1000))
  return [ufile,pfile]

# report times and parameters
str_reportname=str_problem+"_%4.2f_"%(nu)+str_meshtype+"_"+meshname+"_"+str_note
reportfile = open('report_%s_%ds_%dms.csv'%(str_reportname,Time,dt*1000), 'a')
reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
reportwriter.writerow(["a_problem"] + ["meshname"] + ["meshtype"] + ["note"] + ["method"] + ["nu"] + ["time"] + ["dt"] + ["timeToSolve"])
def report():
  reportwriter.writerow([str_problem]+[meshname]+[str_meshtype]+[str_note]+[str_method]+[nu]+[Time]+[dt]+[toc()])

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
    
    #slozeny predpis
    #if t<0.5 : 
    #  v_in.v = sin(pi*t) 
    #else:
    #  v_in.v = 1
    #v_in.v = [sin(pi*t)][1][t<0.5] # by melo dat stejny vysledek
    
    #jednoduchy predpis
    v_in.v = t
    
    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "cg", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    # Save to file
    out[0] << u1
    out[1] << p1
    
    # Move to next time step
    u0.assign(u1)
    t += dt

info("Finished: explicite Chorin method")
report()

info("Initialization of direct method")
str_method="direct"
tic()

# Define function spaces (Taylor-Hood)
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

    # Save to file
    out[0] << u
    out[1] << p
    
    # Move to next time step
    assign(u0,u)
    t += dt

info("Finished: direct method")
report()



