from __future__ import print_function
from dolfin import *

# program used to compute different mesh characteristics for .xml meshes using FEniCS features

# names = ['cyl_c1', 'cyl']
# names = ['cyl_c1', 'cyl_d1', 'cyl_c2', 'cyl_d2', 'cyl_c3', 'cyl_c3o', 'cyl_d3', 'cyl_e3']
# names = ['cyl_c1', 'cyl_c2', 'cyl_c3', 'cyl_c3o', 'cyl_c3o_netgen', 'cyl15_3']
names = ['HYK10']

doPlotQualityHistogram = False
doComputeVolume = False

for meshName in names:
    mesh = Mesh("meshes/" + meshName + ".xml")
    #cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
    #facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")

    #plot(facet_function, interactive=True)

    # mesh = refine(mesh)
    # plot(mesh, interactive=True)

    print("Mesh name: ", meshName, "    ", mesh)
    print("Cell outer circle diameter max/min: ", mesh.hmax(), mesh.hmin())

    volume_min = 1000
    volume_max = 0
    edge_min = 1000
    edge_max = 0
    radius_min = 1000
    radius_max = 0
    count = 0.
    sum = 0.

    for c in cells(mesh):
        v = c.volume()
        if v > volume_max:
            volume_max = v
        if v < volume_min:
            volume_min = v
        r = c.inradius()
        if r > radius_max:
            radius_max = r
        if r < radius_min:
            radius_min = r
    for e in edges(mesh):
        l = e.length()
        sum += l
        count += 1
        if l > edge_max:
            edge_max = l
        if l < edge_min:
            edge_min = l

    avg = sum/count

    print('cell volume max/min:', volume_max, volume_min)
    print('edge length max/min:', edge_max, edge_min)
    print('edge length average:', avg)
    print('cell inner radius max/min', radius_max, radius_min)
    print('min/max outer-inner radius factor (0 worst, 1 best):', MeshQuality.radius_ratio_min_max(mesh))
    if doPlotQualityHistogram:
        exec(MeshQuality.radius_ratio_matplotlib_histogram(mesh, num_bins=100))

    if doComputeVolume:
        V = FunctionSpace(mesh, 'Lagrange', 1)
        volume = assemble(interpolate(Expression('1.0'), V) * dx)
        print('mesh volume:', volume)

    V = VectorFunctionSpace(mesh, 'Lagrange', 2)
    Q = FunctionSpace(mesh, 'Lagrange', 1)
    print('DOFS for P2 velocity:', V.dim())
    print('DOFS for P1 pressure:', Q.dim())

