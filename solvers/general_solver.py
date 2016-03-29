from __future__ import print_function


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

    @staticmethod
    def setup_parser_options(parser):
        pass

    def initialize(self, options):
        pass

    def solve(self, problem):
        pass

    def solve_step(self, dt):
        pass

