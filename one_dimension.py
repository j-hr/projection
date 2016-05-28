from __future__ import print_function

from dolfin import AutoSubDomain, FunctionSpace, DirichletBC, Function, solve, assemble, plot, project, Expression, norm, \
    interpolate
from dolfin.cpp.function import near
from dolfin.cpp.io import interactive
from dolfin.cpp.la import Vector, VectorSpaceBasis, as_backend_type
from dolfin.cpp.mesh import IntervalMesh, FacetFunction
from dolfin.functions import Constant, TrialFunction, TestFunction
import math
from ufl import inner, dot, dx, system, grad, Measure
import os
# NT scheme set to ipcs0 (explicite)
# NT Rotation schemes work well.
# ruining oscillations v=1, l=1, nu> or < 0.01. For nu=0.01 only stable oscilation on outflow.
# for v<=0.1 OK


doSavePlot = True
width = 800
height = 600
dir = '1Dplots_B'
if not os.path.exists(dir):
    os.mkdir(dir)
step = 0

# Set global parameters
timestep = 0.1
time = 0.2

nu_factor = 10.

length = 20.0
v_in = 10.0
nu = Constant(3.7 * nu_factor)
dt = Constant(timestep)
f = Constant(0.0)
mesh_size = 10

# Create meshes and facet function
mesh = IntervalMesh(mesh_size, 0.0, length)
mesh_plot = IntervalMesh(8*mesh_size, 0.0, length)
boundary_parts = FacetFunction('size_t', mesh)
right = AutoSubDomain(lambda x: near(x[0], length))
left = AutoSubDomain(lambda x: near(x[0], 0.0))
right.mark(boundary_parts, 2)
left.mark(boundary_parts, 1)

# Create function spaces
V = FunctionSpace(mesh, 'Lagrange', 2)
# Vplot = FunctionSpace(mesh_plot, 'Lagrange', 1)
Vplot = V
Q = FunctionSpace(mesh, 'Lagrange', 1)

# BC conditions, nullspace
v_in_expr = Constant(v_in)
plt = plot(interpolate(v_in_expr, V), range_min=0., range_max=2*v_in, window_width= width, window_height= height)
plt.write_png('%s/correct' % dir)
# v_in_expr = Expression('(t<1.0)?t*v:v', v=Constant(v_in), t=0.0)
# v_in_expr = Expression('(t<1.0)?(1-cos(pi*t))*v*0.5:v', v=Constant(v_in), t=0.0)
bcp = DirichletBC(Q, Constant(0.0), boundary_parts, 2)
bcu = DirichletBC(V, v_in_expr, boundary_parts, 1)
foo = Function(Q)
null_vec = Vector(foo.vector())
Q.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm('l2')
# print(null_vec.array())
null_space = VectorSpaceBasis([null_vec])

ds = Measure("ds", subdomain_data=boundary_parts)

# Define forms (dont redefine functions used here)
# step 1
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
u_tent = TrialFunction(V)
v = TestFunction(V)
# U_ = 1.5*u0 - 0.5*u1
# nonlinearity = inner(dot(0.5 * (u_tent.dx(0) + u0.dx(0)), U_), v) * dx
# F_tent = (1./dt)*inner(u_tent - u0, v) * dx + nonlinearity\
#     + nu*inner(0.5 * (u_tent.dx(0) + u0.dx(0)), v.dx(0)) * dx + inner(p0.dx(0), v) * dx\
#     - inner(f, v)*dx     # solve to u_
# using explicite scheme: so LHS has interpretation as heat equation, RHS are sources
F_tent = (1./dt)*inner(u_tent - u0, v)*dx + inner(dot(u0.dx(0), u0), v)*dx + nu*inner((u_tent.dx(0) + u0.dx(0)), v.dx(0)) * dx + inner(p0.dx(0), v) * dx\
    - inner(f, v)*dx     # solve to u_
a_tent, L_tent = system(F_tent)
# step 2
u_tent_computed = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)
F_p = inner(grad(p-p0), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx # + 2*p.dx(0)*q*ds(1) # tried to force dp/dn=0 on inflow
# TEST: prescribe Neumann outflow BC
# F_p = inner(grad(p-p0), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx + (1./dt)*(v_in_expr-u_tent_computed)*q*ds(2)
a_p, L_p = system(F_p)
A_p = assemble(a_p)
as_backend_type(A_p).set_nullspace(null_space)
print(A_p.array())
# step 2 rotation
# F_p_rot = inner(grad(p), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx + (1./dt)*(v_in_expr-u_tent_computed)*q*ds(2)
F_p_rot = inner(grad(p), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx
a_p_rot, L_p_rot = system(F_p_rot)
A_p_rot = assemble(a_p_rot)
as_backend_type(A_p_rot).set_nullspace(null_space)
# step 3
p_computed = Function(Q)
u_cor = TrialFunction(V)
F_cor = (1. / dt) * inner(u_cor - u_tent_computed, v) * dx + inner(p_computed.dx(0) - p0.dx(0), v) * dx
a_cor, L_cor = system(F_cor)
# step 3 rotation
u_cor_computed = Function(V)
F_cor_rot = (1. / dt) * inner(u_cor - u_tent_computed, v) * dx + inner(p_computed.dx(0), v) * dx
a_cor_rot, L_cor_rot = system(F_cor_rot)
# step 4 rotation
p_cor_computed = Function(Q)
p_rot = TrialFunction(Q)
F_rot = (p_rot - p0 - p_computed + nu * u_tent_computed.dx(0)) * q * dx
a_rot, L_rot = system(F_rot)

rhs = Function(Q)
rhs_nonlinear = Function(Q)
rhs_visc = Function(Q)
rhs_pressure = Function(Q)
div_u_tent = Function(Vplot)
p_correction = Function(Q)


def solve_cycle(state):
    print('Solving state:', state['name'])
    # u1.assign(state['u_prev'])
    u0.assign(state['u_last'])
    p0.assign(state['pressure'])
    rhs.assign(project(dt*(- u0.dx(0)*u0 - nu*u0.dx(0).dx(0)/2.0 - p0.dx(0)), Q))
    plot(rhs, title='RHS')
    # rhs_nonlinear.assign(project(dt*(- u0.dx(0)*u0), Q))
    # rhs_visc.assign(project(dt*(-nu*u0.dx(0).dx(0)/2.0), Q))
    # rhs_pressure.assign(project(dt*(-p0.dx(0)), Q))
    # plot(rhs_nonlinear, title='RHS nonlin')
    # plot(rhs_visc, title='RHS visc')
    # plot(rhs_pressure, title='RHS pressure')

    solve(a_tent == L_tent, u_tent_computed, bcu)
    if state['rot']:
        if state['null']:
            b = assemble(L_p_rot)
            null_space.orthogonalize(b)
            solve(A_p_rot, p_computed.vector(), b, 'cg')
        else:
            solve(a_p_rot == L_p_rot, p_computed, bcp)
        solve(a_cor_rot == L_cor_rot, u_cor_computed)
        div_u_tent.assign(project(-nu*u_tent_computed.dx(0), Vplot))
        plot(div_u_tent, title=state['name']+'_div u_tent (pressure correction), t = ' +str(t))
        # div_u_tent.assign(project(p0+p_computed-nu*state['p_tent'].dx(0), Q))
        # plot(div_u_tent, title=state['name']+'_RHS (pressure correction), t = ' +str(t))
        solve(a_rot == L_rot, p_cor_computed)
        p_correction.assign(p_cor_computed-p_computed-p0)
        plot(p_correction, title=state['name']+'_(computed pressure correction), t = ' +str(t))
        print('  updating state')
        state['u_prev'].assign(state['u_last'])
        state['u_tent'].assign(u_tent_computed)
        state['u_last'].assign(u_cor_computed)
        state['pressure'].assign(p_cor_computed)
        state['p_tent'].assign(p_computed+p0)
    else:
        if state['null']:
            b = assemble(L_p)
            null_space.orthogonalize(b)
            print('new:', assemble((v_in_expr-u_tent_computed)*ds(2)))
            # plot(interpolate((v_in_expr-u_tent_computed)*ds(2), Q), title='new')
            # print(A_p.array())
            # print(b.array())
            solve(A_p, p_computed.vector(), b, 'gmres')
        else:
            solve(a_p == L_p, p_computed, bcp)
        solve(a_cor == L_cor, u_cor_computed)
        print('  updating state')
        # state['u_prev'].assign(state['u_last'])
        state['u_tent'].assign(u_tent_computed)
        state['u_last'].assign(u_cor_computed)
        state['pressure'].assign(p_computed)


def plot_state(state, t):
    print('Plotting state:', state['name'])
    # plot(state['u_tent'], title=state['name']+'_u_tent')
    # interactive()
    # exit()
    # state['u_tent'].set_allow_extrapolation(True)
    state['u_tent_plot'].assign(interpolate(state['u_tent'], Vplot))
    state['u_last_plot'].assign(interpolate(state['u_last'], Vplot))
    if doSavePlot:
        utp = plot(state['u_tent_plot'], title=state['name']+'_u_tent, t = ' +str(t), range_min=0., range_max=2*v_in, window_width= width, window_height= height)
        utp.write_png('%s/%d_u_tent' % (dir, step))
    else:
        plot(state['u_tent_plot'], title=state['name']+'_u_tent, t = ' +str(t), range_min=-v_in, range_max=3*v_in)
    if doSavePlot:
        ucp = plot(state['u_last_plot'], title=state['name']+'_u_corrected, t = ' +str(t), range_min=0., range_max=2*v_in, window_width= width, window_height= height)
        ucp.write_png('%s/%d_u_cor' % (dir, step))
    else:
        plot(state['u_last_plot'], title=state['name']+'_u_corrected, t = ' +str(t), range_min=-v_in, range_max=3*v_in)
    if state['rot']:
        if doSavePlot:
            # prp = plot(state['p_tent'], title=state['name']+'_pressure, t = ' +str(t), range_min=-1000., range_max=1100., window_width= width, window_height= height)
            prp = plot(state['p_tent'], title=state['name']+'_pressure, t = ' +str(t), range_min=0., range_max=2100., window_width= width, window_height= height)
            prp.write_png('%s/%d_u_pres' % (dir, step))
        else:
            plot(state['p_tent'], title=state['name']+'_tentative pressure, t = ' +str(t))
        if doSavePlot:
            # prp = plot(state['pressure'], title=state['name']+'_corrected pressure, t = ' +str(t), range_min=-1000., range_max=1100., window_width= width, window_height= height)
            prp = plot(state['pressure'], title=state['name']+'_corrected pressure, t = ' +str(t), range_min=0., range_max=2100., window_width= width, window_height= height)
            prp.write_png('%s/%d_pres_cor' % (dir, step))
        else:
            plot(state['pressure'], title=state['name']+'_corrected pressure, t = ' +str(t))
    else:
        if doSavePlot:
            # prp = plot(state['pressure'], title=state['name']+'_pressure, t = ' +str(t), range_min=-1000., range_max=1100., window_width= width, window_height= height)
            prp = plot(state['pressure'], title=state['name']+'_pressure, t = ' +str(t), range_min=0., range_max=2100., window_width= width, window_height= height)
            prp.write_png('%s/%d_pres' % (dir, step))
        else:
            plot(state['pressure'], title=state['name']+'_pressure, t = ' +str(t))


state_B_ = {'name': 'B_', 'u_prev': Function(V), 'u_tent': Function(V), 'u_last': Function(V),
            'pressure': Function(Q), 'p_tent': Function(Q), 'rot': False, 'null': False,
            'u_tent_plot': Function(Vplot), 'u_last_plot': Function(Vplot)}
state_BR = {'name': 'BR', 'u_prev': Function(V), 'u_tent': Function(V), 'u_last': Function(V),
            'pressure': Function(Q), 'p_tent': Function(Q), 'rot': True, 'null': False,
            'u_tent_plot': Function(Vplot), 'u_last_plot': Function(Vplot)}
state_N_ = {'name': 'N_', 'u_prev': Function(V), 'u_tent': Function(V), 'u_last': Function(V),
            'pressure': Function(Q), 'p_tent': Function(Q), 'rot': False, 'null': True,
            'u_tent_plot': Function(Vplot), 'u_last_plot': Function(Vplot)}
state_NR = {'name': 'NR', 'u_prev': Function(V), 'u_tent': Function(V), 'u_last': Function(V),
            'pressure': Function(Q), 'p_tent': Function(Q), 'rot': True, 'null': True,
            'u_tent_plot': Function(Vplot), 'u_last_plot': Function(Vplot)}

# states = [state_B_, state_BR, state_N_, state_NR]
states = [state_B_]

t = timestep
step = 1
while t < time + 1e-6:
    print('t = ', t)
    v_in_expr.t = t
    for state in states:
        solve_cycle(state)
        n = norm(state['u_last'])
        print(n)
        plot_state(state, t)
        interactive()
        if math.isnan(n):
            exit('Failed')
        if n > 5 * v_in * length:
            exit('Norm too big!')
    t += timestep
    step += 1





