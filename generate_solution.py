__author__ = 'jh'

from dolfin import *
import sys
import results
import problem as prb
import argparse

# Import gmsh mesh
parser = argparse.ArgumentParser()
parser.add_argument('mesh', help='Mesh name')
parser.add_argument('dt', help='Time step', type=float)
parser.add_argument('filetype', help='Preffered output type', type=str, choices={'xdmf', 'hdf5'})
parser.add_argument('--nu', help='nu factor', type=float, default=1.0)

args = parser.parse_args()
args.name = 'none'
args.method = 'none'
args.type = 'none'
args.time = 1
args.factor = 1.0
args.error = 'noEC'
args.save = 'noSave'
args.r = False
args.B = False
args.solvers = 'none'
args.prec = 'none'
args.precision = 'none'
args.bc = 'none'
print(args)

problem = prb.Problem('save_solution', args)

meshName = sys.argv[1]
mesh = Mesh("meshes/" + meshName + ".xml")
cell_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_physical_region.xml")
facet_function = MeshFunction("size_t", mesh, "meshes/" + meshName + "_facet_region.xml")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # velocity
PS = FunctionSpace(mesh, "Lagrange", 2)  # partial solution (must be same order as V)

rm = results.ResultsManager(problem, None)
rm.save_solution(args.filetype, PS, V)








