from __future__ import print_function

__author__ = 'jh'

import os
import re
import shutil

regex = re.compile('"f_')


def rewrite_xdmf_files(metadata):
    """changes xdmf vector name "f_something" into something like "problem_namevelocity_diff" """
    # could be done by function.rename('desired name','label') in FEniCS, applied to functions in GeneralProblem
    # but separate function object for every file or renaming before every save would have to be used
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
    try:
        os.remove('temp')
    except OSError:
        pass


def create_scripts(metadata):
    """
    Generates scripts for convenient display of results using ParaView 4.4 Python interface
    """
    abspath = os.path.abspath(os.curdir)
    shutil.copy2('../paraview_scripts/empty.pvsm', 'empty.pvsm')
    if metadata['hasTentativeV']:
        template = open('../paraview_scripts/template_compare_vel_tent_cor.py', 'r')
        out_file = open('compare_vel_tent.py', 'w')
        for line in template:
            fac = 1.0
            if 'factor' in metadata:
                fac = 0.001/metadata['factor']
            line = line.replace('$FACTOR$', str(fac))
            line = line.replace('$DIR$', metadata['dir'])
            line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'velocity_tent.xdmf')
            line = line.replace('$FILENAME2$', metadata['dir']+'/'+metadata['filename_base']+'velocity.xdmf')
            line = line.replace('$VECTORNAME1$', metadata['name']+'velocity_tent')
            line = line.replace('$VECTORNAME2$', metadata['name']+'velocity')
            out_file.write(line)
        template.close()
        out_file.close()
    else:
        template = open('../paraview_scripts/template_velocity.py', 'r')
        out_file = open('show_vel.py', 'w')
        for line in template:
            fac = 1.0
            if 'factor' in metadata:
                fac = 0.001/metadata['factor']
            line = line.replace('$DIR$', metadata['dir'])
            line = line.replace('$FACTOR$', str(fac))
            line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'velocity.xdmf')
            line = line.replace('$VECTORNAME1$', metadata['name']+'velocity')
            out_file.write(line)
        template.close()
        out_file.close()
    if metadata['hasWSS']:
        if metadata['WSSmethod'] == 'expression':
            template = open('../paraview_scripts/template_WSS.py', 'r')
            out_file = open('show_WSS.py', 'w')
            for line in template:
                fac = 1.0
                if 'factor' in metadata:
                    fac = 0.001/metadata['factor']
                line = line.replace('$DIR$', metadata['dir'])
                line = line.replace('$FACTOR$', str(fac))
                line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'wss.xdmf')
                line = line.replace('$VECTORNAME1$', metadata['name']+'wss')
                out_file.write(line)
            template.close()
            out_file.close()
            # WSS norm
            template = open('../paraview_scripts/template_WSSnorm.py', 'r')
            out_file = open('show_WSSnorm.py', 'w')
            for line in template:
                fac = 1.0
                if 'factor' in metadata:
                    fac = 0.001/metadata['factor']
                line = line.replace('$DIR$', metadata['dir'])
                line = line.replace('$FACTOR$', str(fac))
                line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'wss_norm.xdmf')
                line = line.replace('$VECTORNAME1$', metadata['name']+'wss_norm')
                out_file.write(line)
            template.close()
            out_file.close()
        elif metadata['WSSmethod'] == 'integral':
            # WSS norm in DG space
            template = open('../paraview_scripts/template_WSSnormDG.py', 'r')
            out_file = open('show_WSSnormDG.py', 'w')
            for line in template:
                fac = 1.0
                if 'factor' in metadata:
                    fac = 0.001/metadata['factor']
                line = line.replace('$DIR$', metadata['dir'])
                line = line.replace('$FACTOR$', str(fac))
                line = line.replace('$FILENAME1$', metadata['dir']+'/'+metadata['filename_base']+'wss_norm.xdmf')
                line = line.replace('$VECTORNAME1$', metadata['name']+'wss_norm')
                out_file.write(line)
            template.close()
            out_file.close()