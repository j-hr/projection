from __future__ import print_function

from dolfin import parameters


class GeneralSolver:
    def __init__(self, args, tc, metadata):
        self.tc = tc
        self.tc.init_watch('status', 'Reported status.', True)

        self.args = args
        self.metadata = metadata
        self.problem = None

        self.V = None
        self.Q = None
        self.PS = None
        self.D = None

        if args.ffc == 'auto_opt' or args.ffc == 'uflacs_opt':
            parameters["form_compiler"]["optimize"] = True  # NT maybe do nothing with uflacs
        if args.ffc == 'uflacs' or args.ffc == 'uflacs_opt':
            parameters["form_compiler"]["representation"] = "uflacs"

    @staticmethod
    def setup_parser_options(parser):
        parser.add_argument('--ffc', help='Form compiler options', choices=['auto_opt', 'uflacs', 'uflacs_opt', 'auto'], default='auto_opt')

    def initialize(self, options):
        pass

    def solve(self, problem):
        pass

    def solve_step(self, dt):
        pass

