class GeneralSolver:
    def __init__(self, args, tc):
        self.tc = tc
        self.args = args
        self.problem = None

    @staticmethod
    def setup_parser_options(parser):
        pass

    def initialize(self, options):
        pass

    def solve(self, problem, options):
        pass

    def solve_step(self, dt):
        pass

    def update_time(self, actual_time):
        pass