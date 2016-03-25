from __future__ import print_function

import argparse
import sys
import problem as prb
import results

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
# NT depends on analytic solution
parser.add_argument('-e', '--error', help='Error control mode', choices=['doEC', 'noEC', 'test'], default='doEC')
parser.add_argument('-S', '--save', help='Save solution mode', choices=['doSave', 'noSave', 'diff'], default='noSave')
#   doSave: create .xdmf files with velocity, pressure, divergence
#   diff: save also difference vel-sol
#   noSave: do not create .xdmf files with velocity, pressure, divergence
# not resolved (general or problem specific)
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
tc = results.TimeControl()
# TODO init watches. Where? >> move to solver

solver = Solver(args, tc)
problem = Problem(args, tc)

# initialize metadata
metadata = {
    'name': args.name,
    'problem': str(args.problem),
    'solver': str(args.solver),
    'mesh_info': str(problem.mesh),
}

# rm = results.ResultsManager(problem, tc)  IMP results.py code must move into GeneralProblem/specific Problem class

# TODO move to respective files (into init() methods)
print("Problem:       " + args.problem)
print("Solver:       " + args.solver)
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

# solver_options = {'absolute_tolerance': 1e-25, 'relative_tolerance': 1e-12, 'monitor_convergence': True}

# Set parameter values
dt = args.dt
ttime = args.time
print("Time:         %1.0f s\ndt:           %d ms" % (ttime, 1000 * dt))

options = {
    # 'solver_options': solver_options,
    'dt': dt,
    'time': ttime,
}

solver.solve(problem, options)

