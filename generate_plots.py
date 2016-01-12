from __future__ import print_function
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import problem
__author__ = 'jh'

parser = argparse.ArgumentParser()
parser.add_argument('-dir', help='working directory', type=str, default=".")
args = parser.parse_args()
print(args)
os.chdir(args.dir)

if not os.path.exists('plots'):
    os.mkdir('plots')

# reports over seconds =========================================================================
csvfile = open('done_merged_seconds.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';')
seconds_list = next(csvreader)
l = len(seconds_list)
seconds_list = [int(x) for x in seconds_list[4:l]]
print(seconds_list)
data_list = []
data_types = {}
for line in csvreader:
    newdata = [line[0]]+line[2:4]
    for val in line[4:l]:
        newdata.append(float(val))
    data_list.append(newdata)
    print(line[3])
    if not line[3] in data_types:
        data_types[line[3]] = [1, line[2]]
    else:
        data_types[line[3]][0] += 1
csvfile.close()
print(data_types)
print(data_list)
for type_key in data_types.iterkeys():
    print(type_key, data_types[type_key])
    max_value = 0
    empty_plot = True
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, data_types[type_key][0]))
    i = 0
    # TODO use less collors using different line types (write function of N to create list of formatstrings and colors)
    for data in data_list:
        if data[2] == type_key and data[3:]:
            # print(data[3:])
            newmax = max(data[4:])  # first value is not counted for maximum
            if newmax > max_value:
                max_value = newmax
            plt.plot(seconds_list, data[3:], '.-', label=data[0], color=colors[i])
            empty_plot = False
            i += 1
    if not empty_plot:
        # print(max_value)
        plt.xlabel('cycles')
        plt.title(data_types[type_key][1])
        plt.xticks(seconds_list)
        lgd = plt.legend(bbox_to_anchor=(1.4, 1.0))
        plt.axis([min(seconds_list), max(seconds_list), 0, max_value])
        plt.savefig('plots/S_' + type_key + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

# time lines =========================================================================
exit()
# TODO different timesteps? introduce timestep value into .csv
csvfile = open('done_merged_time_lines.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';')
time_list = next(csvreader)
l = len(time_list)
time_list = [float(x) for x in time_list[3:l]]
print(time_list)
data_list = []
data_types = {}
for line in csvreader:
    newdata = [line[0]]+line[2:4]
    for val in line[4:l]:
        newdata.append(float(val))
    data_list.append(newdata)
    if not line[3] in data_types:
        data_types[line[3]] = [1, line[2]]
    else:
        data_types[line[3]][0] += 1
csvfile.close()
print(data_types)
# TODO insert analytic values into right plots ()
for type_key in data_types.iterkeys():
    print(type_key, data_types[type_key])
    max_value = 0
    min_value = 1e16
    empty_plot = True
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, data_types[type_key][0]))
    for data in data_list:
        if data[2] == type_key and data[3:]:
            # print(data[3:])
            newmax = max(data[3:])
            newmin = min(data[3:])
            if newmax > max_value:
                max_value = newmax
            if newmin < max_value:
                min_value = newmin
            plt.plot(time_list, data[3:], '.-', label=data[0])
            empty_plot = False
    if not empty_plot:
        # print(max_value)
        plt.xlabel('cycles')
        plt.title(data_types[type_key][1])
        # plt.xticks(seconds_list)
        plt.legend(bbox_to_anchor=(1.4, 1.0))
        plt.axis([min(time_list), max(time_list), min_value, max_value])
        plt.savefig('plots/TL_' + type_key + '.png')
    plt.close()

# general reports =========================================================================
csvfile = open('done_merged.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';')
header = next(csvreader)
data_list = []
problem_list = []
for line in csvreader:
    # collect names of problems (first five chars of parameter "name")
    problem_name = line[1][0:5]
    if not problem_name in problem_list:
        problem_list.append(problem_name)

    data_list.append([problem_name] + line)
    metadata = problem.load_metadata(line[2])
    print(type(metadata), metadata)

# NT convert data to float!
csvfile.close()

exit()

for i in range(len(header)):
    h = str(header[i])
    if h.startswith('last_cycle'):
        for problem in problem_list:
            print()
            # TODO create convergence plots
