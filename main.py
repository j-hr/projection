from __future__ import print_function

import argparse
import sys
from time_control import TimeControl

# Resolve input arguments===============================================================================================
print(sys.argv)

# IFNEED move time, dt into problem class
parser = argparse.ArgumentParser()
parser.add_argument('problem', help='Which problem to solve', choices=['womersley_cylinder', 'FaC3D_benchmark'])
parser.add_argument('solver', help='Which solver to use', choices=['ipcs1'])
parser.add_argument('mesh', help='Mesh name')
parser.add_argument('time', help='Total time', type=int)
parser.add_argument('dt', help='Time step', type=float)
parser.add_argument('-n', '--name', help='name of this run instance', default='test')
args, remaining = parser.parse_known_args()

exec('from solvers.%s import Solver' % args.solver)
exec('from problems.%s import Problem' % args.problem)

parser = argparse.ArgumentParser()
Solver.setup_parser_options(parser)
Problem.setup_parser_options(parser)
args, remaining = parser.parse_known_args(remaining, args)
print(args)
print('Not parsed:', remaining)

# initialize time control
tc = TimeControl()

# initialize metadata
metadata = {
    'name': args.name,
}

solver = Solver(args, tc, metadata)
problem = Problem(args, tc, metadata)

metadata.update({  # QQ move into problem/solver init
    'problem': str(args.problem),
    'solver': str(args.solver),
    'mesh_info': str(problem.mesh),
})

print("Problem:       " + args.problem)
print("Solver:       " + args.solver)
# TODO move to respective files (into init() methods)
# if args.method in ['chorinExpl', 'ipcs0', 'ipcs1']:
#     problem.d()['hasTentativeVel'] = True
# else:
#     problem.d()['hasTentativeVel'] = False
# if args.method == 'direct':
#     if args.solver != 'default':
#         exit('Parameter solvers should be \'default\' when using direct method.')
# else:
#     if args.solvers == 'krylov':
#         print('Chosen Krylov solvers.')
#     elif args.solvers == 'direct':
#         print('Chosen direct solvers.')

# Set parameter values
dt = args.dt
ttime = args.time
print("Time:         %1.0f s\ndt:           %d ms" % (ttime, 1000 * dt))

metadata.update({
    'dt': dt,
    'dt_ms': int(round(dt*1000)),
    'time': ttime,
})

solver.solve(problem)

