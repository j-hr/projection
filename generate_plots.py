from __future__ import print_function
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import problem as prb
__author__ = 'jh'

parser = argparse.ArgumentParser()
parser.add_argument('-dir', help='working directory', type=str, default=".")
args = parser.parse_args()
print(args)
os.chdir(args.dir)

if not os.path.exists('plots'):
    os.mkdir('plots')


# load general reports =========================================================================
csvfile = open('done_merged.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
header = next(csvreader)[2:]
characteristics = []
ch_index = {}
for item in header:
    if str(item).startswith('last_cycle_') and (str(item).endswith('r') or str(item).endswith('n')):
        characteristics.append(item[11:])
        ch_index[item[11:]] = header.index(item)
print('Used characteristics:', characteristics)
problems = {}
for line in csvreader:
    # collect names of problems (first five chars of parameter "name")
    problem_name = line[0][0:5]
    report_list = []
    for value in line[2:]:
        report_list.append(float(value))
    if problem_name not in problems:
        problems[problem_name] = {line[0]: {'report': report_list, 'md': prb.load_metadata(line[1])}}
    else:
        problems[problem_name][line[0]] = {'report': report_list, 'md': prb.load_metadata(line[1])}
csvfile.close()

print(len(problems.items()))

factors = {1: '0.01', 2: '0.05', 3: '0.1', 4: '0.5', 5: '1.0'}
meshes = range(1, 4)
formats3 = ['x:', '+:', '1:']
formats5 = ['1:', '2:', '.:', 'x:', '+:']
dts = {1: 100, 2: 50, 3: 10, 4: 5, 5: 1}
# create convergence plots =========================================================================
for ch in characteristics:
    # CHARACTERISTIC/DT
    for (f, fs) in factors.iteritems():
        plot_empty = True
        p_number = 0
        for problem in problems.iterkeys():
            # TODO problem comparison
            colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(problems.items())))
            for i in meshes:
                x = []
                y = []
                for (t, dt_ms) in dts.iteritems():
                    name = problem + ('%d%d%d' % (f, i, t))
                    if name in problems[problem]:
                        d = problems[problem][name]
                        y.append(d['report'][ch_index[ch]])
                        x.append(dt_ms)
                if y and sum(y) > 0:
                    plot_empty = False
                    print(ch, ' on mesh ', i, x, y)
                    plt.plot(x, y, formats3[i-1], label=problem + (' on mesh %d' % i), color=colors[p_number])
            p_number += 1
        if not plot_empty:
            plt.xlabel('dt in ms')
            plt.xscale('log')
            plt.yscale('log')
            axis = plt.axis()
            # print(axis)
            axis = [100, 1, axis[2], axis[3]]
            plt.axis(axis)
            plt.title(ch + ' for factor=' + fs)
            lgd = plt.legend(bbox_to_anchor=(1.5, 1.0))
            plt.savefig('plots/C_' + ch + '_f%d' % f + '_CT.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
    # CHARACTERISTIC/H
    for (f, fs) in factors.iteritems():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_empty = True
        p_number = 0
        for problem in problems.iterkeys():
            # TODO problem comparison
            colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(problems.items())))
            for (t, dt_ms) in dts.iteritems():
                x = []
                y = []
                for i in meshes:
                    name = problem + ('%d%d%d' % (f, i, t))
                    if name in problems[problem]:
                        d = problems[problem][name]
                        y.append(d['report'][ch_index[ch]])
                        x.append(d['md']['h'])
                if y and sum(y) > 0:
                    plot_empty = False
                    print(ch, ' dt: ', dt_ms, x, y)
                    ax.plot(x, y, formats5[t-1], label=problem + (' with dt %d' % dt_ms), color=colors[p_number])
            p_number += 1
        if not plot_empty:
            plt.xlabel('h')
            plt.xscale('log')
            plt.yscale('log')
            axis = plt.axis()
            # print(axis)
            axis = [2.3, 0.5, axis[2], axis[3]]
            plt.axis(axis)
            plt.xticks((2.0, 1.0, 0.5), ('2.0', '1.0', '0.5'))
            plt.title(ch + ' for factor=' + fs)
            lgd = plt.legend(bbox_to_anchor=(1.5, 1.0))
            plt.savefig('plots/C_' + ch + '_f%d' % f + '_CS.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

exit()

# reports over seconds =========================================================================
csvfile = open('done_merged_seconds.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
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
    # TODO colors should respect problem configurations
    for data in data_list:
        if data[2] == type_key and data[3:]:
            # print(data[3:])
            newmax = max(data[4:])  # first value is not counted for maximum (it is too high)
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
csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
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

