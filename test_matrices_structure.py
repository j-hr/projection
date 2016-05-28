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

# doSavePlot = True
# width = 800
# height = 600
# dir = 'plots_matrices'
# if not os.path.exists(dir):
#     os.mkdir(dir)
# step = 0

mesh_factor = 5
mesh1D = UnitIntervalMesh(mesh_factor)
mesh2D = UnitSquareMesh(mesh_factor, mesh_factor)
mesh3D = UnitCubeMesh(mesh_factor, mesh_factor, mesh_factor)

polynomial_degree = 1

Q1D = FunctionSpace(mesh1D, 'Lagrange', polynomial_degree)
Q2D = FunctionSpace(mesh2D, 'Lagrange', polynomial_degree)
Q3D = FunctionSpace(mesh3D, 'Lagrange', polynomial_degree)
V2D = VectorFunctionSpace(mesh2D, 'Lagrange', polynomial_degree)
V3D = VectorFunctionSpace(mesh3D, 'Lagrange', polynomial_degree)

cmap = get_cmap('bwr')


def plot_matrix(matrix):
    array = matrix.array()
    max_val = max([abs(float(array.max())), abs(float(array.min()))])
    matshow(array, cmap=cmap, vmin=-max_val, vmax=max_val)
    colorbar()


u = TrialFunction(Q1D)
u_ext = interpolate(Expression("1."), Q1D)
v = TestFunction(Q1D)
a_mass = inner(u, v)*dx
A_mass = assemble(a_mass)
plot_matrix(A_mass)

a_diffusion = inner(u.dx(0), v.dx(0))*dx
A_diffusion = assemble(a_diffusion)
plot_matrix(A_diffusion)

a_convection = inner(dot(u.dx(0), u_ext), v)*dx
A_convection = assemble(a_convection)
plot_matrix(A_convection)

show()

Qspaces = [Q2D, Q3D]
Vspaces = [V2D, V3D]


