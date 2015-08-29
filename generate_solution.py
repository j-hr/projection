__author__ = 'jh'

from dolfin import *
import sys
import results

# Import gmsh mesh
meshName = sys.argv[1]
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

rm = results.ResultsManager()

rm.save_solution(meshName, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], PS, V)








