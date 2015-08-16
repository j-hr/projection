from __future__ import print_function
from dolfin import *
import math
import csv
import sys
from sympy import I, re, im, sqrt, exp, symbols, lambdify, besselj
from scipy.special import jv 
import traceback

#TODO if method = ... change to if hasTentative
#TODO another projection methods (MiroK)
#TODO pulsePrec type = implement solving of stationary problem
    # stationary system solved
    # is result better than pulse0?
    # not working: direct method does not converge for cyl_c1, dt=0.1

#TODO modify onset? (wom)

#Issues
# mesh.hmax() returns strange values

#Notes
# characteristic time for onset ~~ length of pipe/speed of fastest particle = 20(mm) /factor*1081(mm/s) ~~  0.02 s/factor
# characteristic time for dt ~~ hmax/speed of fastest particle = hmax/(factor*1081(mm/s))
#   h = cuberoot(volume/number of cells):
#   c1: 829 cells => h = 1.23 mm => 1.1 ms/factor
#   c2: 6632 cells => h = 0.62 mm => 0.57 ms/factor
#   c3: 53056 cells => h = 0.31 mm => 0.28 ms/factor

tic()
#==Resolve input arguments================================================================================================
print(sys.argv)
nargs = 7
arguments = "method type meshname T dt(minimal: 0.001) factor errorcontrol_start_time(-1 for default starting time, 0 for no errc.) [note to append to result folder]"
if not ((len(sys.argv) == nargs+1) or (len(sys.argv) == nargs+2)):
    exit("Wrong number of arguments. Should be: %d (%s)"%(nargs,arguments))
if (len(sys.argv) == nargs+2):
    str_note = sys.argv[8]
else:
    str_note=""

#choose a method: direct, chorinExpl
str_method=sys.argv[1]
print("Method:       "+str_method)
#choose type of flow:
#   steady - parabolic profile (0.5 s onset)
#Womersley profile (1 s period)
#   pulse0 - u(0)=0
#   pulsePrec - u(0) from precomputed solution (steady Stokes problem)
str_type=sys.argv[2]
print("Problem type: "+str_type)

if sys.argv[7]=="0" :
    doErrControl=False
    print("Error control ommited")
else:
    doErrControl=True
    if sys.argv[7]=="-1" :
        measure_time = 0.5 if str_type=="steady" else 1 # maybe change
    else : measure_time = float(sys.argv[7])
    print("Error control from:       %4.2f s"%(measure_time))

# Set parameter values
dt = float(sys.argv[5])
Time = float(sys.argv[4])
print("Time:         %d s\ndt:           %d ms"%(Time,1000*dt))
factor = float(sys.argv[6]) # default: 1.0
print("Velocity scale factor = %4.2f"%factor)
reynolds = 728.761*factor
print("Computing with Re = %f"%reynolds)

# Import gmsh mesh 
meshname = sys.argv[3]
mesh = Mesh("meshes/"+meshname+".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/"+meshname+"_facet_region.xml")
print("Mesh norm: ",mesh.hmax(),"    ",mesh)
#======================================================================================================================
# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2) # velocity
Q = FunctionSpace(mesh, "Lagrange", 1) # pressure
PS = FunctionSpace(mesh, "Lagrange", 2) # partial solution (must be same order as V)

# fixed parameters (used in analytic solution)
nu = 3.71 # kinematic viscosity
R=5.0 # cylinder radius

#==precomputation of Bessel functions================================================================================
if doErrControl:
    if str_type == "pulse0" or  str_type == "pulsePrec" :      
        temp=toc()
        coefs_mult = [(-11.799 + 0.60076*I),(-11.799 - 0.60076*I),(-26.3758 - 4.65265*I),(-26.3758 + 4.65265*I),(-51.6771 + 27.3133*I),(-51.6771 - 27.3133*I),(-33.1594 - 95.2423*I),(-33.1594 + 95.2423*I)]
        coefs_bes_mult = [(0.000735686 - 0.000528035*I),(0.000735686 + 0.000528035*I),-(0.000814244 - 0.00277126*I),-(0.000814244 + 0.00277126*I),-(0.0110653 - 0.00200668*I),-(0.0110653 + 0.00200668*I),(0.0314408 - 0.0549981*I),(0.0314408 + 0.0549981*I)]
        coefs_bes = [(1.84042 + 1.84042*I),(1.84042 - 1.84042*I),(1.59385 - 1.59385*I),(1.59385 + 1.59385*I),(1.30138 + 1.30138*I),(1.30138 - 1.30138*I),(0.920212 - 0.920212*I),(0.920212 + 0.920212*I)]
        coefs_exp = [-8,8,6,-6,-4,4,2,-2]
        coefs_r_prec = [] # these will be functions in PS
        coefs_i_prec = [] # these will be functions in PS
        c0ex = Expression("factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))",factor=factor)
        c0_prec=(interpolate(c0ex,PS))
        for i in range(8):
            r = symbols('r')
            besRe = re(factor*coefs_mult[i]*(coefs_bes_mult[i]*besselj(0, r*coefs_bes[i]) + 1))
            besIm = im(factor*coefs_mult[i]*(coefs_bes_mult[i]*besselj(0, r*coefs_bes[i]) + 1))
            besRe_lambda = lambdify(r, besRe, ['numpy', {'besselj': jv}])
            besIm_lambda = lambdify(r, besIm, ['numpy', {'besselj': jv}])
            class PartialReSolution(Expression):
                def eval(self, value, x):
                    rad = float(sqrt(x[0]*x[0]+x[1]*x[1])) # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                    value[0] = 0 if near(rad,R) else besRe_lambda(rad) # do not evaluate on boundaries, it's 0
                    #print(value) gives reasonable values
            expr = PartialReSolution()
            coefs_r_prec.append(interpolate(expr,PS))
            class PartialImSolution(Expression):
                def eval(self, value, x):
                    rad = float(sqrt(x[0]*x[0]+x[1]*x[1])) # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
                    value = 0 if near(rad,R) else besIm_lambda(rad) # do not evaluate on boundaries, it's 0
            expr = PartialImSolution()
            coefs_i_prec.append(interpolate(expr,PS))
        #plot(coefs_r_prec[2],title="coefs_r_prec") #reasonable values
        print("Precomputed partial solution functions. Time: %f"%(toc()-temp))

#==Boundary Conditions================================================================================
# boundary parts: 1 walls, 2 inflow, 3 outflow
noslip  = Constant((0.0, 0.0, 0.0))
#TODO pulsePrec - u(0) from precomputed solution (steady Stokes problem)
if str_type == "steady" :
    v_in = Expression(("0.0","0.0","(t<0.5)?((sin(pi*t))*factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))):(factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1])))"),t=0,factor=factor)
elif str_type == "pulse0" or  str_type == "pulsePrec" :      
    r,t = symbols('r t')
    u = factor*(-43.2592*r**2 + (-11.799 + 0.60076*I)*((0.000735686 - 0.000528035*I)*besselj(0, r*(1.84042 + 1.84042*I)) + 1)*exp(-8*I*pi*t) + (-11.799 - 0.60076*I)*((0.000735686 + 0.000528035*I)*besselj(0, r*(1.84042 - 1.84042*I)) + 1)*exp(8*I*pi*t) + (-26.3758 - 4.65265*I)*(-(0.000814244 - 0.00277126*I)*besselj(0, r*(1.59385 - 1.59385*I)) + 1)*exp(6*I*pi*t) + (-26.3758 + 4.65265*I)*(-(0.000814244 + 0.00277126*I)*besselj(0, r*(1.59385 + 1.59385*I)) + 1)*exp(-6*I*pi*t) + (-51.6771 + 27.3133*I)*(-(0.0110653 - 0.00200668*I)*besselj(0, r*(1.30138 + 1.30138*I)) + 1)*exp(-4*I*pi*t) + (-51.6771 - 27.3133*I)*(-(0.0110653 + 0.00200668*I)*besselj(0, r*(1.30138 - 1.30138*I)) + 1)*exp(4*I*pi*t) + (-33.1594 - 95.2423*I)*((0.0314408 - 0.0549981*I)*besselj(0, r*(0.920212 - 0.920212*I)) + 1)*exp(2*I*pi*t) + (-33.1594 + 95.2423*I)*((0.0314408 + 0.0549981*I)*besselj(0, r*(0.920212 + 0.920212*I)) + 1)*exp(-2*I*pi*t) + 1081.48)
    # how this works?
    u_lambda = lambdify([r,t], u, ['numpy', {'besselj': jv}]) 
    class WomersleyProfile(Expression):
        def eval(self, value, x):
            rad = float(sqrt(x[0]*x[0]+x[1]*x[1])) # conversion to float needed, u_lambda (and near) cannot use sympy Float as input
            value[0] = 0
            value[1] = 0
            value[2] = 0 if near(rad,R) else re(u_lambda(rad,t)) # do not evaluate on boundaries, it's 0
        def value_shape(self):
            return (3,)
    v_in=WomersleyProfile()
if str_type == "pulsePrec" :   # computes initial velocity as a solution of steady Stokes problem with input velocity v_in
    begin ("computing initial velocity")
    t = 0 # used in v_in
    
    # Define function spaces (Taylor-Hood)
    W = MixedFunctionSpace([V,Q])
    bc0    = DirichletBC(W.sub(0), noslip, facet_function,1)
    inflow = DirichletBC(W.sub(0), v_in, facet_function,2)
    # Collect boundary conditions
    bcu = [inflow,bc0]
    # Define unknown and test function(s) NS
    v, q = TestFunctions(W)
    u, p = TrialFunctions(W)
    w = Function(W)
    # Define fields
    n = FacetNormal(mesh)
    I = Identity(u.geometric_dimension())    # Identity tensor
    x = SpatialCoordinate(mesh)
    # Define steady part of the equation
    def T(u):
        return -p*I + 2.0*nu*sym(grad(u))

    # Define variational forms
    F = (inner(T(u), grad(v)) - q*div(u))*dx
    solve(lhs(F)==rhs(F), w, bcu) # why this way?
    # Extract solutions:
    (u_prec, p_prec) = w.split()
    end()
    
    #plot(u_prec, mode = "glyphs", title="steady solution", interactive=True)
    #exit()

#==Analytic solution====================================================================================
if doErrControl :
    temp=toc()
    if str_type == "steady" :
        global solution
        solution = interpolate(Expression(("0.0","0.0","factor*(1081.48-43.2592*(x[0]*x[0]+x[1]*x[1]))"),factor=factor),V)
        print("Prepared analytic solution. Time: %f"%(toc()-temp))
    elif (str_type == "pulse0") or (str_type == "pulsePrec") :
        def assembleSolution(t): # returns Womersley solution for time t   
            temp=toc()
            solution = Function(V)
            dofs2 = V.sub(2).dofmap().dofs() #gives field of indices corresponding to z axis
            solution.assign(Constant(("0.0","0.0","0.0")))
            solution.vector()[dofs2] = c0_prec.vector() # parabolic part of solution
            for i in range(8): # add modes of Womersley solution
                solution.vector()[dofs2] += cos(coefs_exp[i]*pi*t)*coefs_r_prec[i].vector().array()
                solution.vector()[dofs2] += -sin(coefs_exp[i]*pi*t)*coefs_i_prec[i].vector().array()
            print("Assembled analytic solution. Time: %f"%(toc()-temp))
            return solution
    #plot(assembleSolution(0.2), mode = "glyphs", title="solution")
    #interactive()
    #exit()
    #save solution
    #f=File("solution.xdmf")
    #t = dt
    #s= Function(V)
    #while t < Time + DOLFIN_EPS:
        #print("t = ", t)
        #s.assign(assembleSolution(t))
        #f << s
        #t+=dt
    #exit()    

#==Output settings====================================================================================
# Create files for storing solution
str_dir_name = str_note+"results_"+str_type+"_"+str_method+"_"+meshname+"_factor%4.2f_%ds_%dms"%(factor,Time,dt*1000)
ufile = File(str_dir_name+"/velocity.xdmf")
dfile = File(str_dir_name+"/divergence.xdmf") # maybe just compute norm
pfile = File(str_dir_name+"/pressure.xdmf")
if str_method=="chorinExpl" : 
    u2file = File(str_dir_name+"/velocity_tent.xdmf")
    d2file = File(str_dir_name+"/div_tent.xdmf") # maybe just compute norm

# method for saving divergence (ensuring, that it will be one timeline in Paraview)
D = FunctionSpace(mesh, "Lagrange", 1)
divu = Function(D)
def savediv(field,divfile):
    temp=toc()
    divu.assign(project(div(field), D))
    divfile << divu
    print("Computed and saved divergence. Time: %f"%(toc()-temp))

# method for saving velocity (ensuring, that it will be one timeline in Paraview)
vel = Function(V)
def savevel(field,velfile):
    temp=toc()
    vel.assign(field)
    velfile << vel
    print("Saved solution. Time: %f"%(toc()-temp))

div_u = []
div_u2 = []
def computeDiv(divlist,velocity):
    temp=toc()
    divlist.append(norm(velocity, 'Hdiv0'))
    print("Computed norm of divergence. Time: %f"%(toc()-temp))

def reportFail(inst):
    print("Runtime error:", sys.exc_info()[1])
    print("Traceback:")
    traceback.print_tb(sys.exc_info()[2])
    f = open(str_note+"_factor%4.2f_step_%dms_failed_at_%5.3f.report"%(factor,dt*1000,t),"w")
    f.write(traceback.format_exc())
    f.close()

#==Error control====================================================================================
time_erc=0  # total time spent on measuring error
if doErrControl : 
    timelist=[] # list of times, when error is  measured (used in report)
    err_u = []
    err_u2 = []
    def computeErr(erlist,velocity,t):
        if len(timelist)==0 or (timelist[-1] < round(t,3)) : # add only once per time step
            timelist.append(round(t,3)) # round timestep to 0.001
        temp=toc()
        if str_type == "steady" :
            #erlist.append(pow(errornorm(velocity, solution, norm_type='l2', degree_rise=0),2)) # slower, more reliable
            erlist.append(assemble(inner(velocity-solution,velocity-solution)*dx)) # faster
        elif (str_type == "pulse0") or (str_type == "pulsePrec") :
            #erlist.append(pow(errornorm(velocity, assembleSolution(t), norm_type='l2', degree_rise=0),2)) # degree rise?
            sol = assembleSolution(t) # name must be different than solution - solution must be treated as global
            erlist.append(assemble(inner(velocity-sol,velocity-sol)*dx)) # faster
        global time_erc
        terc = toc() - temp
        time_erc += terc
        print("Computed errornorm. Time: %f, Total: %f"%(terc,time_erc))

#==Explicite Chorin method====================================================================================
if str_method=="chorinExpl" :
    info("Initialization of explicite Chorin method")
    tic()

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
    if str_type == "pulsePrec" :   
        assign(u0,u_prec)
    savevel(u0,ufile)
    savevel(u0,u2file)
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
        try:
            solve(A1, u1.vector(), b1, "gmres", "default")
        except RuntimeError as inst:
            reportFail(inst)
            exit()
        savevel(u1,u2file)
        if doErrControl and round(t,3)>=measure_time : computeErr(err_u2,u1,t)
        computeDiv(div_u2,u1)
        savediv(u1,d2file)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        try:
            solve(A2, p1.vector(), b2, "cg", prec)
        except RuntimeError as inst:
            reportFail(inst)
            exit()
        pfile << p1
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        try:
            solve(A3, u1.vector(), b3, "gmres", "default")
        except RuntimeError as inst:
            reportFail(inst)
            exit()
        savevel(u1,ufile)
        if doErrControl and round(t,3)>=measure_time : computeErr(err_u,u1,t)
        computeDiv(div_u,u1)
        savediv(u1,dfile)
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
    W = MixedFunctionSpace([V,Q])
    bc0    = DirichletBC(W.sub(0), noslip, facet_function,1)
    inflow = DirichletBC(W.sub(0), v_in, facet_function,2)
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
    
    # to assign solution in space W.sub(0) to Function(V) we need FunctionAssigner (cannot be assigned directly)
    fa=FunctionAssigner(V,W.sub(0))
    velSp = Function(V)
    
    #Define fields for time dependent case
    u0 = Function(V) #velocity from previous time step
    if str_type == "pulsePrec" :   
        assign(u0,u_prec)
    savevel(u0,ufile)


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

    #prm = NS_solver.parameters
    #prm['newton_solver']['absolute_tolerance'] = 1E-08
    #prm['newton_solver']['relative_tolerance'] = 1E-08
    #prm['newton_solver']['maximum_iterations'] = 45
    #prm['newton_solver']['relaxation_parameter'] = 1.0
    #prm['newton_solver']['linear_solver'] = 'mumps' #or petsc,mumps

    # Time-stepping
    info("Running of direct method")
    t = dt
    while t < Time + DOLFIN_EPS:
        print("t = ", t)
        
        v_in.t=t
        
        # Compute
        begin("Solving NS ....")
        try:
            NS_solver.solve()
        except RuntimeError as inst:
            reportFail(inst)
            exit()
        end()

        # Extract solutions:
        (u, p) = w.split()

        fa.assign(velSp,u) # we are assigning twice (now and inside savevel), but it works with one method savevel for direct and projection (we can split savevel to save one assign)
        savevel(velSp,ufile)
        savediv(u,dfile)
        computeDiv(div_u,u)
        pfile << p
        if doErrControl and round(t,3)>=measure_time : computeErr(err_u,u,t)

        # Move to next time step
        assign(u0,u)
        t += dt

    info("Finished: direct method")
    total=toc()

#==Report====================================================================================
total_err_u = 0 
total_err_u2 = 0
avg_err_u = 0 
avg_err_u2 = 0 
if doErrControl : 
    total_err_u = math.sqrt(sum(err_u))
    total_err_u2 = math.sqrt(sum(err_u2))
    avg_err_u = total_err_u/math.sqrt(len(timelist))
    avg_err_u2 = total_err_u2/math.sqrt(len(timelist)) 
    err_u = [math.sqrt(i) for i in err_u]
    err_u2 = [math.sqrt(i)   for i in err_u2]
    # report of errornorm for individual timesteps
    with open(str_dir_name+"/report_err.csv", 'w') as reportfile:
        reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
        reportwriter.writerow(["name"]+["time"]+timelist)
        reportwriter.writerow([str_note]+["corrected"]+err_u)
        if str_method=="chorinExpl" : reportwriter.writerow([str_note]+["tentative"]+err_u2)

# report of norm of div for individual timesteps
with open(str_dir_name+"/report_div.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow([str_note]+["corrected"]+div_u)
    if str_method=="chorinExpl" : reportwriter.writerow([str_note]+["tentative"]+div_u2)

# report without header
with open(str_dir_name+"/report.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow(["pipe_test"]+[str_note]+[str_type]+[str_method]+[meshname]+[mesh]+[factor]+[Time]+[dt]+[total-time_erc]+[time_erc]+[time_erc]+[total_err_u]+[total_err_u2]+[avg_err_u]+[avg_err_u2])

# report with header
with open(str_dir_name+"/report_h.csv", 'w') as reportfile:
    reportwriter = csv.writer(reportfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
    reportwriter.writerow(["problem"] + ["name"]+["type"] + ["method"] + ["meshname"] + ["mesh"]+["factor"]+ ["time"] + ["dt"] + ["timeToSolve"] + ["timeToComputeErr"] + ["toterrVel"] + ["toterrVelTent"]+["avg_err_u"]+["avg_err_u2"])
    reportwriter.writerow(["pipe_test"]+[str_note]+[str_type]+[str_method]+[meshname]+[mesh]+[factor]+[Time]+[dt]+[total-time_erc]+[time_erc]+[total_err_u]+[total_err_u2]+[avg_err_u]+[avg_err_u2])

# create file showing all was done well
f = open(str_note+"_factor%4.2f_step_%dms_OK.report"%(factor,dt*1000),"w")
f.close()

