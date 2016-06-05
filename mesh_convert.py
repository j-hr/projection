from dolfin import *

# to convert xml mesh with facet function to hdf5
meshes = ['cyl_c3']

for m in meshes:
    mesh = Mesh('meshes/' + m + '.xml')
    cell_function = MeshFunction("size_t", mesh, "meshes/" + m + "_physical_region.xml")
    facet_function = MeshFunction("size_t", mesh, "meshes/" + m + "_facet_region.xml")

    f = HDF5File(mpi_comm_world(), 'meshes/'+m+'.hdf5', 'w')
    f.write(mesh, 'mesh')
    f.write(facet_function, 'facet_function')
    f.close()

    f = HDF5File(mpi_comm_world(), 'meshes/'+m+'.hdf5', 'r')
    mesh = Mesh()
    f.read(mesh, 'mesh', False)
    facet_function = MeshFunction("size_t", mesh)
    f.read(facet_function, 'facet_function')
    # mesh = Mesh('meshes/' + m + '.hdf5')
    plot(mesh)
    plot(facet_function)
    interactive()

