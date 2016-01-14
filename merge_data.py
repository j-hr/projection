from __future__ import print_function

__author__ = 'jh'

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dir', help='working directory', type=str, default=".")
args = parser.parse_args()
print(args)
os.chdir(args.dir)
dtToI = {'0.1': 1, '0.05': 2, '0.01': 3, '0.005': 4, '0.001': 5}

header = ''
header_seconds = ''
headers_time_lines = ['not_used', '', '', '', '', '']
search_dirs = []
merged_report = open('temp_merged.csv', 'w')
merged_report_seconds = open('temp_merged_seconds.csv', 'w')
merged_report_time_lines = ['not_used']
for i in range(1, 6):
    temp = open('temp_merged_time_lines%d.csv' % i, 'w')
    merged_report_time_lines.append(temp)
for f in os.listdir(os.path.abspath(os.curdir)):
    # print(f)
    if f[0] != '.':
        # print(os.getcwd())
        # print(os.stat(f))
        if os.path.isdir(f):
            search_dirs.append(f)
# print(search_dirs)
for sd in search_dirs:
    # print(sd)
    for f in os.listdir(os.path.abspath(sd)):
        if not os.path.isdir(f):
            if f == 'report_h.csv':
                filepath = os.path.join(os.path.abspath(sd), f)
                if header == '':
                    with open(filepath, 'r') as openfile:
                        header = openfile.readline()
                        merged_report.write(header)
                        data_line = openfile.readline()
                else:
                    with open(filepath, 'r') as openfile:
                        newHeader = openfile.readline()
                        data_line = openfile.readline()
                    if newHeader != header:
                        print('Inconsistent headlines!')
                        exit()
                merged_report.write(data_line)
                print(filepath)
            if f == 'report_seconds.csv':
                filepath = os.path.join(os.path.abspath(sd), f)
                openfile = open(filepath, 'r')
                if header_seconds == '':
                    header_seconds = openfile.readline()
                    merged_report_seconds.write(header_seconds)
                else:
                    openfile.readline()
                for line in openfile:
                    merged_report_seconds.write(line)
                print(filepath)
            if f == 'report_time_lines.csv':
                filepath = os.path.join(os.path.abspath(sd), f)
                openfile = open(filepath, 'r')
                first = openfile.readline()
                i = dtToI[first.split(';', 4)[3]]
                if headers_time_lines[i] == '':
                    headers_time_lines[i] = first
                    merged_report_time_lines[i].write(headers_time_lines[i])
                for line in openfile:
                    merged_report_time_lines[i].write(line)
                print(filepath)
merged_report.close()
merged_report_seconds.close()
for i in range(1, 6):
    merged_report_time_lines[i].close()
    os.rename('temp_merged_time_lines%d.csv' % i, 'done_merged_time_lines%d.csv' % i)
os.rename('temp_merged.csv', 'done_merged.csv')
os.rename('temp_merged_seconds.csv', 'done_merged_seconds.csv')



