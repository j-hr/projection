from __future__ import print_function
from dolfin import *
import sys
__author__ = 'jh'

meshName = sys.argv[1]
factor = 1.0
print('Mesh: '+meshName)
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

f = HDF5File(mpi_comm_world(), 'precomputed_'+meshName+'.hdf5', 'r')
print(f)
print(f.has_dataset('imag0'))
fce = Function(PS)
f.read(fce, 'imag1')
plot(fce, interactive=True)
