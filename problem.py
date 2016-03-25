from __future__ import print_function
import cPickle
__author__ = 'jh'

meshToH = {
    'cyl_c1': 2.24,
    'cyl_c2': 1.07,
    'cyl_c3': 0.53,
    'cyl_c3o': 0.37,
    'cyl_c3o_netgen': 0.35,
    'cyl_d1': 2.18,
    'cyl_d2': 1.07,
    'cyl_d3': 0.53,
    'cyl_e3': 0.39,
    'cyl15_3': 0.12,
}


class Problem:
    def __init__(self, problem_name, args):
        self.metadata = {
            'problem': problem_name,
            'name': args.name,
            'type': args.type,
            'method': args.method,
            'mesh': args.mesh,
            'h': meshToH[args.mesh],
            'solvers': args.solvers,
            'solver precision': args.precision,
            'preconditioner pressure solver': args.prec,
            'pressureBC': args.bc,
            'scale factor': args.factor,
            'cycles': args.time,
            'nu': 3.71 * args.nu,
            'nu_factor': args.nu,
            'dt': args.dt,
            'dt_ms': int(round(args.dt*1000)),
            'rotation scheme': args.r,
            'no3rdBC': args.B,
            'factor': args.factor
        }
        self.ECoption = args.error
        self.saveoption = args.save

    def d(self):
        return self.metadata

    def get_metadata_to_save(self):
        return str(cPickle.dumps(self.metadata)).replace('\n', '$')

def load_metadata(code):
    return cPickle.loads(code.replace('$', '\n'))
