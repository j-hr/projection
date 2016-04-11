from __future__ import print_function

__author__ = 'jh'

import os
import re

regex = re.compile('"f_')


def rewrite_xdmf_files(metadata):
    """changes xdmf vector name "f_something" into something like "IBC_I111velocity_diff" """
    os.chdir(metadata['dir'])
    for f in os.listdir('.'):
        if f.endswith('xdmf'):
            name = f[5:-5]
            print('Rewriting file: %-40s new vector name:' % f, name)
            os.rename(f, 'temp')
            try:
                reader = open('temp', 'r')
                writer = open(f, 'w')
                for line in reader:
                    if re.search(regex, line):
                        s = line.split('\"')
                        newline = line.replace(s[1], name)
                    else:
                        newline = line
                    writer.write(newline)
                reader.close()
                writer.close()
            except IOError:
                print('IOError:', f)
    os.remove('temp')


def create_scripts(metadata):
    abspath = os.path.abspath(os.curdir)
    template = open('../paraview_scripts/template_compare_vel_tent_cor.py', 'r')
    out_file = open('compareveldiff.py', 'w')
    for line in template:
        fac = 1.0
        if 'factor' in metadata:
            fac = 0.001/metadata['factor']
        line = line.replace('$FACTOR$', str(fac))
        line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'velocity_tent.xdmf')
        line = line.replace('$FILENAME2$', metadata['dir']+'/'+metadata['filename_base']+'velocity.xdmf')
        line = line.replace('$VECTORNAME1$', metadata['name']+'velocity_tent')
        line = line.replace('$VECTORNAME2$', metadata['name']+'velocity')
        out_file.write(line)
    template.close()
    out_file.close()

