from __future__ import print_function

__author__ = 'jh'

from dolfin import *
import sys
import womersleyBC

# this program tests if solution assembled from precomputed partial solutions (and saved to hdf5 using
# generate_solution.py) is really the same as solution generated by womersleyBC.WomersleyProfile() (ineffective)

# Import gmsh mesh
mesh_name = sys.argv[1]
mesh = Mesh("meshes/" + mesh_name + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + mesh_name + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + mesh_name + "_facet_region.xml")

V = VectorFunctionSpace(mesh, 'Lagrange', 2)

f = HDF5File(mpi_comm_world(), 'solution_'+mesh_name+'.hdf5', 'r')

precomputed_profile = Function(V)
t = 0

sympy_expr = womersleyBC.WomersleyProfile(1.0)

# To plot individual time step and the difference:
#
# t = 190
#
# if f.has_dataset('sol'+str(t)):
#     sympy_expr.t = float(t)/1000.0
#     sympy_profile = interpolate(sympy_expr, V)
#     f.read(precomputed_profile, 'sol'+str(t))
#     plot(sympy_profile, mode = "glyphs", title = 'sympy', interactive = True)
#     plot(precomputed_profile, mode = "glyphs", title = 'precomputed', interactive = True)
#     plot(precomputed_profile-sympy_profile, mode = "glyphs", interactive = True)
#     print(errornorm(precomputed_profile, sympy_profile, norm_type='l2', degree_rise=0))
# exit()

norms = []

t = 0
while t <= 1000:
    if f.has_dataset('sol'+str(t)):
        print('reading data set: sol', t, 'comparing to sympy_profile, t = ', float(t)/1000.0)
        sympy_expr.t = float(t)/1000.0
        sympy_profile = interpolate(sympy_expr, V)
        f.read(precomputed_profile, 'sol'+str(t))
        norms.append(errornorm(precomputed_profile, sympy_profile, norm_type='l2', degree_rise=0))
    t += 10

print(norms)
print('Sum: ',sum(norms))