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
    os.chdir(metadata['dir'])

