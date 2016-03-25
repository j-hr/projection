class GeneralProblem:
    def __init__(self, args, tc):
        self.tc = tc
        self.args = args
        self.nu_factor = args.nu

    @staticmethod
    def setup_parser_options(parser):
        parser.add_argument('--nu', help='kinematic viscosity factor', type=float, default=1.0)

    def initialize(self, *args):
        pass

    def get_boundary_conditions(self, *args):
        pass

    def get_initial_conditions(self, *args):
        pass
