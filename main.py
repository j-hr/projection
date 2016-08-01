from __future__ import print_function

import argparse
import sys
from dolfin import set_log_level, INFO, DEBUG, parameters
from dolfin.cpp.common import mpi_comm_world, MPI, info
from dolfin.cpp.la import PETScOptions

import postprocessing
from time_control import TimeControl

# Resolve input arguments===============================================================================================
# IFNEED move time, dt, mesh into problem class
parser = argparse.ArgumentParser()
parser.add_argument('problem', help='Which problem to solve', choices=['womersley_cylinder', 'steady_cylinder',
                                                                       'FaC3D_benchmark', 'real'])
parser.add_argument('solver', help='Which solver to use', choices=['ipcs1', 'direct'])
parser.add_argument('mesh', help='Mesh name')
parser.add_argument('time', help='Total time', type=float)
parser.add_argument('dt', help='Time step', type=float)
parser.add_argument('-n', '--name', help='name of this run instance', default='test')
parser.add_argument('--out', help='Which processors in parallel should print output?', choices=['all', 'main'],
                    default='main')
args, remaining = parser.parse_known_args()

# additional output
PETScOptions.set('ksp_view')  # shows info about used PETSc Solver and preconditioner
# if args.solver == 'ipcs1':
    # PETScOptions.set('log_summary')

# Paralell run initialization
comm = mpi_comm_world()
rank = MPI.rank(comm)
# parameters["std_out_all_processes"] = False   # print only rank==0 process output
# parameters["ghost_mode"] = "shared_facet"     # may be needed for operating DG elements in parallel

# allows output using info() for the main process only
if rank == 0 or args.out == 'all':
    set_log_level(INFO)
else:
    set_log_level(INFO + 1)

info('Running on %d processor(s).' % MPI.size(comm))

if MPI.size(comm) > 1 and args.problem == 'womersley_cylinder':
    info('Womersley cylinder problem is not runnable in parallel due to method of construction of analytic solution,'
         ' which is used to describe boundary conditions.')  # the change of mesh format would be also needed
    exit()

# dynamically import selected solver and problem files
exec('from solvers.%s import Solver' % args.solver)
exec('from problems.%s import Problem' % args.problem)

# setup and parse problem- and solver-specific command-line arguments
parser = argparse.ArgumentParser()
Solver.setup_parser_options(parser)
Problem.setup_parser_options(parser)
args, remaining = parser.parse_known_args(remaining, args)
info(str(args))
info('Not parsed:'+str(remaining))

# initialize time control
tc = TimeControl()

# initialize metadata (collection of information about particular run of this program)
metadata = {
    'name': args.name,
}

solver = Solver(args, tc, metadata)
problem = Problem(args, tc, metadata)

info("Problem:      " + args.problem)
info("Solver:       " + args.solver)

# Set parameter values
dt = args.dt
ttime = args.time
info('Time:         %f s' % ttime)
info('dt:           %f s' % dt)
metadata.update({
    'problem': str(args.problem),
    'solver': str(args.solver),
    'mesh_info': str(problem.mesh),
    'dt': dt,
    'time': ttime,
})

r = solver.solve(problem)
out = {0: 'Solver finished correctly.', 1: 'Solver failed or solution diverged, exception caught.'}
info(out.get(r, 'UNCAUGHT ERROR IN SOLVE METHOD'))

if rank == 0 and problem.doSave:
    info('Post-processing')
    postprocessing.rewrite_xdmf_files(metadata)
    postprocessing.create_scripts(metadata)
