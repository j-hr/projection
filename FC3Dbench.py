from dolfin import *
import mshr

set_log_level(WARNING)

# mesh factors, doporuceno 10
factor = 50

# Define domain
H = 0.41
L = 2.5

zero = Point(0.0, 0.0, 0.0)
corner = Point(L, H, H)
box = mshr.Box(zero, corner)

cyl_offset = 0.5
right = Point(cyl_offset, 0.2, H)
left = Point(cyl_offset, 0.2, 0)
radius = 0.05
cylinder = mshr.Cylinder(left, right, radius, radius)
geometry = box - cylinder

# Build mesh
mesh = mshr.generate_mesh(geometry, factor)

plot(mesh, title="mesh", interactive=True)
exit()

# Construct facet markers
bndry = FacetFunction("size_t", mesh)
for f in facets(mesh):
    mp = f.midpoint()
    if near(mp[0], 0.0): bndry[f] = 1  # inflow
    elif near(mp[0], L): bndry[f] = 2  # outflow
    elif near(mp[1], 0.0) or near(mp[1], W): bndry[f] = 3  # walls
    elif mp.distance(center) <= radius:      bndry[f] = 5  # cylinder
