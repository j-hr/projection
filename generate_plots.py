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
print('Loaded characteristics:', characteristics)
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
# default characteristics: ['CE_L2r', 'CE_H1r', 'CE_H1wr', 'PEn', 'TE_L2r', 'TE_H1r', 'TE_H1wr', 'PTEn', 'FEr']
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

# reports over seconds - load data =========================================================================
characteristics_seconds = []
csvfile = open('done_merged_seconds.csv', 'r')
csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
seconds_list = next(csvreader)
l = len(seconds_list)
seconds_list = [int(x) for x in seconds_list[3:l]]
print(seconds_list)
for line in csvreader:
    data_place = problems[line[0][0:5]][line[0]]
    data_list = [float(x) for x in line[3:l]]
    if 'seconds' in data_place:
        data_place['seconds'][line[2]] = data_list
    else:
        data_place['seconds'] = {line[2]: data_list}
    if not line[2] in characteristics_seconds:
        characteristics_seconds.append(line[2])
csvfile.close()
print('Characteristics for second-lines:', characteristics_seconds)

# reports over seconds - generate plots =========================================================================

# QQ what to plot?

# for type_key in data_types.iterkeys():
#     print(type_key, data_types[type_key])
#     max_value = 0
#     empty_plot = True
#     colors = plt.get_cmap('jet')(np.linspace(0, 1.0, data_types[type_key][0]))
#     i = 0
#     # TODO use less collors using different line types (write function of N to create list of formatstrings and colors)
#     # TODO colors should respect problem configurations
#     for data in data_list:
#         if data[2] == type_key and data[3:]:
#             # print(data[3:])
#             newmax = max(data[4:])  # first value is not counted for maximum (it is too high)
#             if newmax > max_value:
#                 max_value = newmax
#             plt.plot(seconds_list, data[3:], '.-', label=data[0], color=colors[i])
#             empty_plot = False
#             i += 1
#     if not empty_plot:
#         # print(max_value)
#         plt.xlabel('cycles')
#         plt.title(data_types[type_key][1])
#         plt.xticks(seconds_list)
#         lgd = plt.legend(bbox_to_anchor=(1.4, 1.0))
#         plt.axis([min(seconds_list), max(seconds_list), 0, max_value])
#         plt.savefig('plots/S_' + type_key + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
#     plt.close()

# time lines - load data =========================================================================
# TODO different timesteps? introduce timestep value into .csv
characteristics_timelines = []
times = {}
for i in range(1, 6):
    csvfile = open('done_merged_time_lines%d.csv' % i, 'r')
    csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
    line = next(csvreader)
    times[i] = [float(x) for x in line[3:]]
    print(times[i])
    for line in csvreader:
        data_place = problems[line[0][0:5]][line[0]]
        data_list = [float(x) for x in line[3:]]
        if 'timelines' in data_place:
            data_place['timelines'][line[2]] = data_list
        else:
            data_place['timelines'] = {line[2]: data_list}
        if not line[2] in characteristics_timelines:
            characteristics_timelines.append(line[2])
    csvfile.close()
print('Characteristics for time-lines:', characteristics_timelines)
# last time characteristics:
# 'AVN_L2', 'AVN_H1',
# 'CE_L2', 'CE_L2s', 'CE_L2n', 'CE_L2r',
# 'DT', 'DTs',
# 'APG', 'APGs', 'APGn',
# 'PGEA', 'PGEAs', 'PGEAn',
# 'CE_H1', 'CE_H1s', 'CE_H1r',
# 'PG', 'PGs', 'PGn',
# 'AF',
# 'CE_H1w', 'CE_H1ws', 'CE_H1wr',
# 'APN',
# 'TE_L2', 'TE_L2s', 'TE_L2n', 'TE_L2r',
# 'TE_H1', 'TE_H1s', 'TE_H1r',
# 'DC', 'DCs',
# 'TE_H1w', 'TE_H1ws', 'TE_H1wr',
# 'PE', 'PEs', 'PEn',
# 'PGE', 'PGEs', 'PGEn',
# 'FE', 'FEr',
# 'AVN_H1w'

# define plots:

# plot: one parameter, one problem, one factor, 3 meshes, 5 dts
plot1 = ['CE_L2', 'CE_L2r', 'CE_H1', 'CE_H1r', 'CE_H1w', 'CE_H1wr', 'TE_L2', 'TE_L2r', 'TE_H1', 'TE_H1t', 'TE_H1w',
         'TE_H1wr', 'FE', 'FEr', 'PE', 'PEn', 'PGE', 'PGEn',]

# QQ which to use?  >> let choose shorter set

# same plot with analytic value
plot2 = {'PG': 'APG'}  # NT: iplement ad hoc

# QQ other plots?

formats3 = ['-', '--', ':']
dtToSteps = {1: 10, 2: 20, 3: 100, 4: 200, 5: 1000}  # time steps per second

for ch in plot1:
    if ch in characteristics_timelines:
        for (f, fs) in factors.iteritems():
            for problem in problems.iterkeys():
                plot_empty = True
                colors = plt.get_cmap('jet')(np.linspace(0, 1.0, 5))
                max_ = 0
                min_ = 1e16
                max_lc = 0
                min_lc = 1e16
                for i in meshes:
                    for (t, dt_ms) in dts.iteritems():
                        name = problem + ('%d%d%d' % (f, i, t))
                        x = []
                        y = []
                        if name in problems[problem]:
                            d = problems[problem][name]
                            y = d['timelines'][ch]
                            x = times[t]
                            newmin = min(y[dtToSteps[t]:])
                            if newmin < min_:
                                min_ = newmin
                            newmax = max(y[dtToSteps[t]:])
                            if newmax > max_:
                                max_ = newmax
                            newmin = min(y[(5*dtToSteps[t]):])  # NT programmed only for 6 cycles!
                            if newmin < min_lc:
                                min_lc = newmin
                            newmax = max(y[(5*dtToSteps[t]):])  # NT programmed only for 6 cycles!
                            if newmax > max_lc:
                                max_lc = newmax
                        if y:
                            plot_empty = False
                            print(ch, ' on mesh ', i)
                            plt.plot(x, y, formats3[i-1], label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                     color=colors[t-1])
                if not plot_empty:
                    plt.xlabel('time')
                    axis = [0, 6, min_, max_]  # NT programmed only for 6 cycles!
                    plt.axis(axis)
                    plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                    lgd = plt.legend(bbox_to_anchor=(1.7, 1.0))
                    plt.savefig('plots/TL1_' + problem + '_' + ch + '_f%d' % f + '.png', bbox_extra_artists=(lgd,),
                                bbox_inches='tight', dpi=300)
                    # save same plot only for last cycle
                    axis = [5, 6, min_lc, max_lc]  # NT programmed only for 6 cycles!
                    plt.axis(axis)
                    plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                    lgd = plt.legend(bbox_to_anchor=(1.7, 1.0))
                    plt.savefig('plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'lc.png', bbox_extra_artists=(lgd,),
                                bbox_inches='tight', dpi=300)
                plt.close()

# the same plots, now splitted for meshes
for ch in plot1:
    if ch in characteristics_timelines:
        for (f, fs) in factors.iteritems():
            for problem in problems.iterkeys():
                for i in meshes:
                    plot_empty = True
                    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, 5))
                    max_ = 0
                    min_ = 1e16
                    max_lc = 0
                    min_lc = 1e16
                    for (t, dt_ms) in dts.iteritems():
                        name = problem + ('%d%d%d' % (f, i, t))
                        x = []
                        y = []
                        if name in problems[problem]:
                            d = problems[problem][name]
                            y = d['timelines'][ch]
                            x = times[t]
                            newmin = min(y[dtToSteps[t]:])
                            if newmin < min_:
                                min_ = newmin
                            newmax = max(y[dtToSteps[t]:])
                            if newmax > max_:
                                max_ = newmax
                            newmin = min(y[(5*dtToSteps[t]):])  # NT programmed only for 6 cycles!
                            if newmin < min_lc:
                                min_lc = newmin
                            newmax = max(y[(5*dtToSteps[t]):])  # NT programmed only for 6 cycles!
                            if newmax > max_lc:
                                max_lc = newmax
                        if y:
                            plot_empty = False
                            print(ch, ' on mesh ', i)
                            plt.plot(x, y, '-', label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                     color=colors[t-1])
                    if not plot_empty:
                        plt.xlabel('time')
                        axis = [0, 6, min_, max_]  # NT programmed only for 6 cycles!
                        plt.axis(axis)
                        plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                        lgd = plt.legend(bbox_to_anchor=(1.7, 1.0))
                        plt.savefig('plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'm%d' % i + '.png',
                                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
                        axis = [5, 6, min_lc, max_lc]  # NT programmed only for 6 cycles!
                        plt.axis(axis)
                        plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                        lgd = plt.legend(bbox_to_anchor=(1.7, 1.0))
                        plt.savefig('plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'm%d' % i + 'lc.png',
                                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
                    plt.close()
