from __future__ import print_function

from dolfin import AutoSubDomain, FunctionSpace, DirichletBC, Function, solve, assemble, plot, project, Expression, norm, \
    interpolate, VectorFunctionSpace
from dolfin.cpp.function import near
from dolfin.cpp.io import interactive
from dolfin.cpp.la import Vector, VectorSpaceBasis, as_backend_type
from dolfin.cpp.mesh import IntervalMesh, FacetFunction, UnitSquareMesh, UnitCubeMesh, UnitIntervalMesh
from dolfin.functions import Constant, TrialFunction, TestFunction
import math
from ufl import inner, dot, dx, system, grad, Measure
import os
from matplotlib import cm
from matplotlib.cm import get_cmap
from matplotlib.pyplot import matshow, show, colorbar
import matplotlib.pyplot as plt

# doSavePlot = True
# width = 800
# height = 600
# dir = 'plots_matrices'
# if not os.path.exists(dir):
#     os.mkdir(dir)
# step = 0

mesh_factor = 1
mesh1D = UnitIntervalMesh(mesh_factor)
mesh2D = UnitSquareMesh(mesh_factor, mesh_factor)
mesh3D = UnitCubeMesh(mesh_factor, mesh_factor, mesh_factor)

# plot(mesh3D)
# print(mesh3D.num_cells())
# interactive()
# exit()

polynomial_degree_V = 2
polynomial_degree_P = 1

Q1D = FunctionSpace(mesh1D, 'Lagrange', polynomial_degree_P)
Q2D = FunctionSpace(mesh2D, 'Lagrange', polynomial_degree_P)
Q3D = FunctionSpace(mesh3D, 'Lagrange', polynomial_degree_P)
V2D = VectorFunctionSpace(mesh2D, 'Lagrange', polynomial_degree_V)
V3D = VectorFunctionSpace(mesh3D, 'Lagrange', polynomial_degree_V)

cmap = get_cmap('bwr')

fignum = 0


def plot_matrix(matrix, name):
    global fignum
    fignum += 1
    fig = plt.figure(fignum)
    array = matrix.array()
    max_val = max([abs(float(array.max())), abs(float(array.min()))])
    matshow(array, cmap=cmap, vmin=-max_val, vmax=max_val, fignum=fignum)
    colorbar()
    fig.set_label(name)

u = TrialFunction(Q1D)
u_ext = interpolate(Expression("1."), Q1D)
v = TestFunction(Q1D)
a_mass = inner(u, v)*dx
A_mass = assemble(a_mass)
# plot_matrix(A_mass)

a_diffusion_1D = inner(u.dx(0), v.dx(0))*dx
A_diffusion_1D = assemble(a_diffusion_1D)
# plot_matrix(A_diffusion_1D)
# plot_matrix(A_diffusion_1D)

a_convection_1D = inner(dot(u.dx(0), u_ext), v)*dx
A_convection_1D = assemble(a_convection_1D)
# plot_matrix(A_convection_1D)

u_exts = [interpolate(Expression(('1.', '1.')), V2D), interpolate(Expression(('1.', '1.', '1.')), V3D)]
Qspaces = [Q2D, Q3D]
Vspaces = [V2D, V3D]
names = ['2D', '3D']

for i in [0, 1]:
    Qspace = Qspaces[i]
    Vspace = Vspaces[i]
    u = TrialFunction(Vspace)
    v = TestFunction(Vspace)
    a_mass = inner(u, v)*dx
    A_mass = assemble(a_mass)
    a_diffusion = inner(grad(u), grad(v))*dx
    A_diffusion = assemble(a_diffusion)
    a_convection = inner(dot(grad(u), u_exts[i]), v)*dx
    A_convection = assemble(a_convection)
    plot_matrix(A_mass, 'mass' + names[i])
    plot_matrix(A_diffusion, 'diff' + names[i])
    plot_matrix(A_convection, 'conv' + names[i])

show()


