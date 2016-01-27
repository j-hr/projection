from __future__ import print_function
import csv, os, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dir', help='working directory', type=str, default=".")
args = parser.parse_args()
print(args)
os.chdir(args.dir)

f_str = {1: '0.01', 2: '0.05', 3: '0.1', 4: '0.5', 5: '1.0'}
data_tables = {}

# LOAD DATA
for f in glob.glob('*.report'):
    s = f.replace('.report', '')
    problem_name = s[0:5]
    factor = int(s[5])
    mesh = int(s[6])
    dt = int(s[7])
    ok = s.endswith('OK')

    if not problem_name in data_tables:
        data_tables[problem_name] = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

    if not mesh in data_tables[problem_name][factor]:
        data_tables[problem_name][factor][mesh] = {}

    data_tables[problem_name][factor][mesh][dt] = 'OK' if ok else 'F '

out = open('tables.csv', 'w')
writer = csv.writer(out, delimiter=';', escapechar='|', quoting=csv.QUOTE_NONE)
for p_name, p in data_tables.iteritems():
    writer.writerow([p_name])
    for factor, data in p.iteritems():
        writer.writerow(['', 'factor = '+f_str[factor]])
        writer.writerow(['', '', '100 ms', '50 ms', '10 ms', '5 ms', '1 ms'])
        for mesh in range(1, 4):
            line = ['', 'mesh %d' % mesh]
            if mesh in data:
                for t in range(1, 6):
                    if t in data[mesh]:
                        line.append(data[mesh][t])
                    else:
                        line.append('N ')
            writer.writerow(line)
        writer.writerow('')


