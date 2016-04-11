from __future__ import print_function

from dolfin import AutoSubDomain, FunctionSpace, DirichletBC, Function, solve, assemble, plot, project
from dolfin.cpp.function import near
from dolfin.cpp.io import interactive
from dolfin.cpp.la import Vector, VectorSpaceBasis, as_backend_type
from dolfin.cpp.mesh import IntervalMesh, FacetFunction
from dolfin.functions import Constant, TrialFunction, TestFunction

# Set global parameters
from ufl import inner, dot, dx, system, grad

timestep = 0.05
time = 1.0

nu = Constant(0.001)
dt = Constant(timestep)
f = Constant(0.0)
mesh_size = 20


# Create meshes and facet function
mesh = IntervalMesh(mesh_size, 0, 1)
mesh_plot = IntervalMesh(4*mesh_size, 0, 1)
boundary_parts = FacetFunction('size_t', mesh)
right = AutoSubDomain(lambda x: near(x[0], 1.0))
left = AutoSubDomain(lambda x: near(x[0], 0.0))
right.mark(boundary_parts, 2)
left.mark(boundary_parts, 1)

# Create function spaces
V = FunctionSpace(mesh, 'Lagrange', 2)
Vplot = FunctionSpace(mesh_plot, 'Lagrange', 1)
Q = FunctionSpace(mesh, 'Lagrange', 1)

# BC conditions, nullspace
bcp = DirichletBC(Q, Constant(0.0), boundary_parts, 2)
bcu = DirichletBC(V, Constant(1.0), boundary_parts, 1)
foo = Function(Q)
null_vec = Vector(foo.vector())
Q.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm('l2')
null_space = VectorSpaceBasis([null_vec])

# Define forms (dont redefine functions used here)
# step 1
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
u_tent = TrialFunction(V)
v = TestFunction(V)
U_ = 1.5*u0 - 0.5*u1
nonlinearity = inner(dot(0.5 * (u_tent.dx(0) + u0.dx(0)), U_), v) * dx
F_tent = (1./dt)*inner(u_tent - u0, v) * dx + nonlinearity\
    + nu*inner((u_tent.dx(0) + u0.dx(0)), v.dx(0)) * dx + inner(p0.dx(0), v) * dx\
    - inner(f, v)*dx     # solve to u_
a_tent, L_tent = system(F_tent)
# step 2
u_tent_computed = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)
F_p = inner(grad(p-p0), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx
a_p, L_p = system(F_p)
A_p = assemble(a_p)
as_backend_type(A_p).set_nullspace(null_space)
# step 2 rotation
F_p_rot = inner(grad(p), grad(q))*dx + (1./dt)*u_tent_computed.dx(0)*q*dx
a_p_rot, L_p_rot = system(F_p_rot)
A_p_rot = assemble(a_p_rot)
as_backend_type(A_p_rot).set_nullspace(null_space)
# step 3
p_computed = Function(Q)
u_cor = TrialFunction(V)
F_cor = (1. / dt) * inner(u_cor - u_tent_computed, v) * dx + inner(p_computed.dx(0) - p0, v) * dx
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


def solve_cycle(state):
    print('Solving state:', state['name'])
    u1.assign(state['u_prev'])
    u0.assign(state['u_last'])
    p0.assign(state['pressure'])
    solve(a_tent == L_tent, u_tent_computed, bcu)
    if state['rot']:
        if state['null']:
            b = assemble(L_p_rot)
            null_space.orthogonalize(b)
            solve(A_p_rot, p_computed, b)
        else:
            solve(a_p_rot == L_p_rot, p_computed, bcp)
        solve(a_cor_rot == L_cor_rot, u_cor_computed, bcu)
        solve(a_rot == L_rot, p_cor_computed)
        print('  updating state')
        state['u_prev'].assign(state['u_last'])
        state['u_tent'].assign(u_tent_computed)
        state['u_last'].assign(u_cor_computed)
        state['pressure'].assign(p_cor_computed)
        state['p_tent'].assign(p_computed)
    else:
        if state['null']:
            b = assemble(L_p)
            null_space.orthogonalize(b)
            solve(A_p, p_computed, b)
        else:
            solve(a_p == L_p, p_computed, bcp)
        solve(a_cor == L_cor, u_cor_computed, bcu)
        print('  updating state')
        state['u_prev'].assign(state['u_last'])
        state['u_tent'].assign(u_tent_computed)
        state['u_last'].assign(u_cor_computed)
        state['pressure'].assign(p_computed)


def plot_state(state):
    print('Plotting state:', state['name'])
    state['u_tent_plot'].assign(project(state['u_tent'], Vplot))
    state['u_last_plot'].assign(project(state['u_last'], Vplot))
    plot(state['u_tent_plot'], title=state['name']+'_u_tent')
    plot(state['u_last_plot'], title=state['name']+'_u_corrected')
    if state['rot']:
        plot(state['p_tent'], title=state['name']+'_tentative pressure')
        plot(state['pressure'], title=state['name']+'_corrected pressure')
    else:
        plot(state['pressure'], title=state['name']+'_pressure')


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
while t < time + 1e-6:
    print('t = ', t)
    for state in states:
        solve_cycle(state)
        plot_state(state)
    interactive()
    t += timestep





