from __future__ import print_function
import os, sys, traceback


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

        # stopping criteria (for relative H1 velocity error norm) (if known)
        self.divergence_treshold = 10

    @staticmethod
    def setup_parser_options(parser):
        pass

    def report_fail(self, t):
        print("Runtime error:", sys.exc_info()[1])
        print("Traceback:")
        traceback.print_tb(sys.exc_info()[2])
        f = open(self.metadata['name'] + "_failed_at_%5.3f.report" % t, "w")
        f.write(traceback.format_exc())
        f.close()
        self.remove_status_file()

    def write_status_file(self, t, last_status_functional, status_functional_str):
        self.tc.start('status')
        f = open(self.metadata['name'] + ".run", "w")
        progress = t/self.metadata['time']
        f.write('t = %5.3f (dt=%3dms)\nprogress = %3.0f %%\n%s = %5.3f\n' %
                (t, self.metadata['dt_ms'], 100*progress, status_functional_str, last_status_functional))
        f.close()
        self.tc.end('status')

    def remove_status_file(self):
        os.remove(self.metadata['name'] + ".run")

    def initialize(self, options):
        pass

    def solve(self, problem):
        pass

    def solve_step(self, dt):
        pass

