from __future__ import print_function
from dolfin import *

meshName = 'HYK'

mesh = Mesh("meshes/" + meshName + ".xml")
tdim = mesh.topology().dim()
mesh.init(tdim-1, tdim)  # needed for facet.exterior()
normal = FacetNormal(mesh)
print("Mesh name: ", meshName, "    ", mesh)
print("Mesh norm max: ", mesh.hmax())
print("Mesh norm min: ", mesh.hmin())
edge_min = 1000
edge_max = 0
for e in edges(mesh):
    l = e.length()
    if l > edge_max:
        edge_max = l
    if l < edge_min:
        edge_min = l
print('edge length max/min:', edge_max, edge_min)
tol = edge_min/10.


# returns number of plane the given point is in
def point_in_subdomain(p):
    if abs(p[1]+13.6391) < tol:
        return 2  # inflow
    elif abs(-0.838444*p[0]+0.544988*p[2]+12.558) < tol:
        return 3  # outflow
    elif abs(0.0933764*p[0] - 0.933764*p[1] - 0.345493*p[2] + 10.5996) < tol:
        return 4  # inflow
    elif abs(-p[0]+20.6585) < tol:
        return 5  # outflow

# Create boundary markers
facet_function = FacetFunction("size_t", mesh)
for f_mesh in facets(mesh):
    if f_mesh.exterior():
        if all(point_in_subdomain(v.point()) == 2 for v in vertices(f_mesh)):
            facet_function[f_mesh] = 2
        elif all(point_in_subdomain(v.point()) == 3 for v in vertices(f_mesh)):
            facet_function[f_mesh] = 3
        elif all(point_in_subdomain(v.point()) == 4 for v in vertices(f_mesh)):
            facet_function[f_mesh] = 4
        elif all(point_in_subdomain(v.point()) == 5 for v in vertices(f_mesh)):
            facet_function[f_mesh] = 5
        else:
            facet_function[f_mesh] = 1   # wall

f_mesh = HDF5File(mpi_comm_world(), 'meshes/' + meshName + '.hdf5', 'w')
f_mesh.write(mesh, 'mesh')
f_mesh.write(facet_function, 'facet_function')
f_mesh.close()

ds3 = Measure("ds", subdomain_id=3, subdomain_data=facet_function)
ds5 = Measure("ds", subdomain_id=5, subdomain_data=facet_function)
ds2 = Measure("ds", subdomain_id=2, subdomain_data=facet_function)
ds4 = Measure("ds", subdomain_id=4, subdomain_data=facet_function)

V = FunctionSpace(mesh, 'Lagrange', 1)
one = interpolate(Expression('1.'), V)

volume = assemble(one*dx)
S2 = assemble(one*ds2)
S3 = assemble(one*ds3)
S4 = assemble(one*ds4)
S5 = assemble(one*ds5)


f_ini = open('meshes/' + meshName + '.ini', 'w')
f_ini.write('in 2 4\n')
f_ini.write('out 3 5\n')
f_ini.write('wall 1\n')
f_ini.write('volume %f\n' % volume)
f_ini.write('2 %f\n' % S2)
f_ini.write('3 %f\n' % S3)
f_ini.write('4 %f\n' % S4)
f_ini.write('5 %f\n' % S5)
f_ini.close()
