from __future__ import print_function
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import problem as prb
import math
__author__ = 'jh'

# global variables
characteristics = []
ch_index = {}
problems = {}
problem_list = []
characteristics_seconds = []
characteristics_timelines = []
times = {}

color_set = 'Set1'   # e. g. 'Set1', 'Dark2', 'brg', 'jet'
dpi = 300
# factors = {1: '0.01', 2: '0.05', 3: '0.1', 4: '0.5', 5: '1.0'}
factors = {1: '0.01'}
meshes = range(1, 4)
dts = {1: 100, 2: 50, 3: 10, 4: 5, 5: 1}
dtToSteps = {1: 10, 2: 20, 3: 100, 4: 200, 5: 1000}  # time steps per second


def savefig(figure, path, lgd, **kwargs):
    figure.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=dpi, **kwargs)


def minmax(minval, maxval, newmin, newmax):
    if newmin < minval:
        minval = newmin
    if newmax > maxval:
        maxval = newmax
    return minval, maxval


def color(color_number, color_range):
    return plt.get_cmap(color_set)(np.linspace(0, 1.0, color_range))[color_number]


# load general reports =========================================================================
def load_general_reports():
    csvfile = open('done_merged.csv', 'r')
    csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
    header = next(csvreader)[2:]
    for item in header:
        if str(item).startswith('last_cycle_') and (str(item).endswith('r') or str(item).endswith('n')):
            characteristics.append(item[11:])
            ch_index[item[11:]] = header.index(item)
        if str(item).startswith('totalTimeHours'):
            characteristics.append('time')
            ch_index['time'] = header.index(item)
    print('Loaded characteristics:', characteristics)
    for line in csvreader:
        # collect names of problems (first five chars of parameter "name")
        problem_name = line[0][0:5]
        report_list = []
        for value in line[2:]:
            report_list.append(float(value))
        if problem_name not in problems:
            problems[problem_name] = {line[0]: {'report': report_list, 'md': prb.load_metadata(line[1])}}
            problem_list.append(problem_name)
        else:
            problems[problem_name][line[0]] = {'report': report_list, 'md': prb.load_metadata(line[1])}
    csvfile.close()
    print('Loaded problems:', problem_list)


# reports over seconds - load data =========================================================================
def load_seconds_data():
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


# time lines - load data =========================================================================
def load_timelines_data():
    for i in range(1, 6):
        csvfile = open('done_merged_time_lines%d.csv' % i, 'r')
        csvreader = csv.reader(csvfile, delimiter=';', escapechar='|')
        try:
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
        except StopIteration:
            print('  File empty.')
        csvfile.close()
    print('Characteristics for time-lines:', characteristics_timelines)


# create convergence plots =========================================================================
# default characteristics: ['time', 'CE_L2r', 'CE_H1r', 'CE_H1wr', 'PEn', 'TE_L2r', 'TE_H1r', 'TE_H1wr', 'PTEn', 'FEr']
def create_convergence_plots():
    formats3 = ['x-.', '+--', '1-']
    line_widths3 = [3.0, 2.0, 1.5]
    formats5 = ['1:', '2:', '3:', 'x:', '+:']
    line_width5 = 3.0
    marker_size = 10.0
    marker_edge_width = 2.0
    for ch in characteristics:
        # CHARACTERISTIC/DT
        print('Convergence in time:', ch)
        for (f, fs) in factors.iteritems():
            print('  f =', fs)
            # create figures and subplot instances
            plots = {}
            fig_idx = 1
            for (plot_name, plot_set) in c_plot.iteritems():
                plots[plot_name] = {}
                plots[plot_name]['fig'] = plt.figure(fig_idx)
                plots[plot_name]['spl'] = plots[plot_name]['fig'].add_subplot('111')
                plots[plot_name]['set'] = plot_set
                plots[plot_name]['empty'] = True
                fig_idx += 1

            for problem in problems.iterkeys():
                print('    problem =', problem)
                for i in meshes:
                    x = []
                    y = []
                    for (t, dt_ms) in dts.iteritems():
                        name = problem + ('%d%d%d' % (f, i, t))
                        if name in problems[problem]:
                            d = problems[problem][name]
                            value = d['report'][ch_index[ch]]
                            if value < 1e12:  # do not plot values from diverging problems
                                y.append(value)
                                x.append(dt_ms)
                    if y and sum(y) > 0:
                        print('      ', ch, ' on mesh ', i, x, y)
                        for (plot_name, plot) in plots.iteritems():
                            if problem in plot['set']:
                                plot['empty'] = False
                                idx = plot['set'].index(problem)
                                rng = len(plot['set'])
                                plot['spl'].plot(x, y, formats3[i-1], label=problem + (' on mesh %d' % i),
                                                 color=color(idx, rng), lw=line_widths3[i - 1], ms=marker_size,
                                                 mew=marker_edge_width)
            for (plot_name, plot) in plots.iteritems():
                if not plot['empty']:
                    axes = plot['fig'].axes[0]
                    plot['spl'].set_xlabel('dt in ms')
                    axes.set_xscale('log')
                    axes.set_yscale('log')
                    axis = axes.axis()
                    axis = [105, 0.95, axis[2], axis[3]]
                    axes.axis(axis)
                    axes.set_title(plot_name + ' ' + ch + ' for factor=' + fs)
                    lgd = axes.legend(bbox_to_anchor=(1.5, 1.0))
                    savefig(plot['fig'], 'plots/C_' + plot_name + '_' + ch + '_f%d' % f + '_CT.png', lgd)
            for idx in range(1, fig_idx):
                plt.figure(idx)
                plt.close()
        # CHARACTERISTIC/H
        print('Convergence in space:', ch)
        for (f, fs) in factors.iteritems():
            print('  f =', fs)
            # create figures and subplot instances
            plots = {}
            fig_idx = 1
            for (plot_name, plot_set) in c_plot.iteritems():
                plots[plot_name] = {}
                plots[plot_name]['fig'] = plt.figure(fig_idx)
                plots[plot_name]['spl'] = plots[plot_name]['fig'].add_subplot('111')
                plots[plot_name]['set'] = plot_set
                plots[plot_name]['empty'] = True
                fig_idx += 1

            for problem in problems.iterkeys():
                print('    problem =', problem)
                for (t, dt_ms) in dts.iteritems():
                    x = []
                    y = []
                    for i in meshes:
                        name = problem + ('%d%d%d' % (f, i, t))
                        if name in problems[problem]:
                            d = problems[problem][name]
                            value = d['report'][ch_index[ch]]
                            if value < 1e12:  # do not plot values from diverging problems
                                y.append(value)
                                x.append(d['md']['h'])
                    if y and sum(y) > 0:
                        print('        ', ch, ' dt: ', dt_ms, x, y)
                        for (plot_name, plot) in plots.iteritems():
                            if problem in plot['set']:
                                plot['empty'] = False
                                idx = plot['set'].index(problem)
                                rng = len(plot['set'])
                                plot['spl'].plot(x, y, formats5[t-1], label=problem + (' with dt %d' % dt_ms),
                                                 color=color(idx, rng), lw=line_width5, ms=marker_size,
                                                 mew=marker_edge_width)
            for (plot_name, plot) in plots.iteritems():
                if not plot['empty']:
                    axes = plot['fig'].axes[0]
                    plot['spl'].set_xlabel('h')
                    axes.set_xscale('log')
                    axes.set_yscale('log')
                    axis = axes.axis()
                    axis = [2.3, 0.5, axis[2], axis[3]]
                    axes.axis(axis)
                    axes.set_xticks([2.0, 1.0, 0.5])
                    axes.set_xticklabels(['2.0', '1.0', '0.5'])
                    axes.set_title(plot_name + ' ' + ch + ' for factor=' + fs)
                    lgd = axes.legend(bbox_to_anchor=(1.5, 1.0))
                    savefig(plot['fig'], 'plots/C_' + plot_name + '_' + ch + '_f%d' % f + '_CS.png', lgd)
            for idx in range(1, fig_idx):
                plt.figure(idx)
                plt.close()


def create_timelines_plots():
    formats3 = ['-.', '--', '-']
    line_widths3 = [2.0, 1.5, 1.0]
    for ch in plot1:
        if ch in characteristics_timelines:
            print(ch)
            for (f, fs) in factors.iteritems():
                print('  f =', fs)
                for problem in problems.iterkeys():
                    print('    problem =', problem)
                    plot_empty = True
                    colors = plt.get_cmap(color_set)(np.linspace(0, 1.0, 5))
                    max_ = 0
                    min_ = 1e16
                    max_l = 0
                    min_l = 1e16
                    for i in meshes:
                        for (t, dt_ms) in dts.iteritems():
                            name = problem + ('%d%d%d' % (f, i, t))
                            x = []
                            y = []
                            if name in problems[problem]:
                                d = problems[problem][name]
                                y = d['timelines'][ch]
                                x = times[t]
                                print('      ' + name + ' data sizes (x, y):', len(x), len(y), 'expected', d['md']['cycles']*dtToSteps[t])
                                min_, max_ = minmax(min_, max_, min(y[dtToSteps[t]:]), max(y[dtToSteps[t]:]))
                                min_l, max_l = minmax(min_l, max_l, min(y[(d['md']['cycles']-1)*dtToSteps[t]:]), max(y[(d['md']['cycles']-1)*dtToSteps[t]:]))
                            if y:
                                plot_empty = False
                                plt.subplot(121)
                                plt.plot(x, y, formats3[i-1], label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                         color=colors[t-1], lw=line_widths3[i-1])
                                plt.subplot(122)
                                plt.yscale('log')
                                plt.plot(x, y, formats3[i-1], label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                         color=colors[t-1], lw=line_widths3[i-1])
                    if not plot_empty:
                        plt.xlabel('time')
                        if ch.endswith('r') and max_ > 10:
                            max_ = 10
                        axis = [0, d['md']['cycles'], min_, max_]
                        min_log = math.pow(10, math.floor(math.log10(min_)))
                        plt.subplot(121)
                        plt.axis(axis)
                        plt.subplot(122)
                        axis = [0, d['md']['cycles'], min_log, max_]
                        plt.axis(axis)
                        plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                        lgd = plt.legend(bbox_to_anchor=(2.5, 1.0))
                        savefig(plt.gcf(), 'plots/TL1_' + problem + '_' + ch + '_f%d' % f + '.png', lgd)
                        # save same plot only for last cycle
                        axis = [d['md']['cycles']-1, d['md']['cycles'], min_l, max_l]
                        min_log = math.pow(10, math.floor(math.log10(min_l)))
                        plt.subplot(121)
                        plt.axis(axis)
                        plt.subplot(122)
                        axis = [d['md']['cycles']-1, d['md']['cycles'], min_log, max_l]
                        plt.axis(axis)
                        plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                        lgd = plt.legend(bbox_to_anchor=(2.5, 1.0))
                        savefig(plt.gcf(), 'plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'lc.png', lgd)
                    plt.close()
    exit()
    # the same plots, now splitted for meshes
    for ch in plot1:
        if ch in characteristics_timelines:
            print(ch)
            for (f, fs) in factors.iteritems():
                print('  f =', fs)
                for problem in problems.iterkeys():
                    print('    problem =', problem)
                    for i in meshes:
                        plot_empty = True
                        colors = plt.get_cmap(color_set)(np.linspace(0, 1.0, 5))
                        max_ = 0
                        min_ = 1e16
                        max_l = 0
                        min_l = 1e16
                        for (t, dt_ms) in dts.iteritems():
                            name = problem + ('%d%d%d' % (f, i, t))
                            x = []
                            y = []
                            if name in problems[problem]:
                                d = problems[problem][name]
                                y = d['timelines'][ch]
                                x = times[t]
                                print('      ' + name + ' data sizes (x, y):', len(x), len(y), 'expected', d['md']['cycles']*dtToSteps[t])
                                min_, max_ = minmax(min_, max_, min(y[dtToSteps[t]:]), max(y[dtToSteps[t]:]))
                                min_l, max_l = minmax(min_l, max_l, min(y[(d['md']['cycles']-1)*dtToSteps[t]:]), max(y[(d['md']['cycles']-1)*dtToSteps[t]:]))
                            if y:
                                plot_empty = False
                                plt.subplot(121)
                                plt.plot(x, y, '-', label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                         color=colors[t-1], lw=1.0)
                                plt.subplot(122)
                                plt.yscale('log')
                                plt.plot(x, y, '-', label=problem + (' on mesh %d' % i) + 'with dt=%d ms' % dt_ms,
                                         color=colors[t-1], lw=1.0)
                        if not plot_empty:
                            plt.subplot(121)
                            plt.xlabel('time')
                            if ch.endswith('r') and max_ > 10:
                                max_ = 10
                            axis = [0, d['md']['cycles'], min_, max_]
                            plt.axis(axis)
                            plt.subplot(122)
                            axis = [0, d['md']['cycles'], 0, max_]
                            plt.axis(axis)
                            plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                            lgd = plt.legend(bbox_to_anchor=(2.5, 1.0))
                            savefig(plt.gcf(), 'plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'm%d' % i + '.png', lgd)
                            axis = [d['md']['cycles']-1, d['md']['cycles'], min_l, max_l]
                            plt.subplot(121)
                            plt.axis(axis)
                            plt.subplot(122)
                            axis = [d['md']['cycles']-1, d['md']['cycles'], 0, max_l]
                            plt.axis(axis)
                            plt.title(ch + ' for ' + problem + ' for factor=' + fs)
                            lgd = plt.legend(bbox_to_anchor=(2.5, 1.0))
                            savefig(plt.gcf(), 'plots/TL1_' + problem + '_' + ch + '_f%d' % f + 'm%d' % i + 'lc.png', lgd)
                        plt.close()


# MAIN code ========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-dir', help='working directory', type=str, default=".")
args = parser.parse_args()
print(args)
os.chdir(args.dir)

if not os.path.exists('plots'):
    os.mkdir('plots')

load_general_reports()
load_seconds_data()
load_timelines_data()
# define convergence plots
# TODO compare corrected and tentative velocity
# TODO compare 'PEn' amd 'PTEn' for rotation schemes
# characteristics = ['time', 'CE_L2r', 'CE_H1r', 'CE_H1wr', 'PEn', 'TE_L2r', 'TE_H1r', 'TE_H1wr', 'FEr']
characteristics = ['time', 'CE_L2r', 'CE_H1r', 'CE_H1wr', 'PEn', 'FEr']
# QQ is it possible to merge characteristics comparison to current code?
c_plot = {'all': problem_list,
          'normal vs no3bc': ['IBC_I', 'IBCbI'],
          'normal vs rotation': ['IBC_I', 'IBCRI'],
          'rotation vs rot+no3bc': ['IBCRI', 'IBCrI']}
for oneplot in c_plot['all']:
    c_plot[oneplot] = [oneplot]

# create_convergence_plots()

# last time characteristics for timelines:
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
# plot1 = ['CE_H1r']
plot1 = ['CE_L2r', 'CE_H1r', 'CE_H1wr', 'FEr', 'FE', 'FNE', 'FSEr', 'FSE', 'PEn', 'AF', 'AFN', 'AFS']
# plot1 = ['CE_L2', 'CE_L2r', 'CE_H1', 'CE_H1r', 'CE_H1w', 'CE_H1wr', 'TE_L2', 'TE_L2r', 'TE_H1', 'TE_H1t', 'TE_H1w',
#          'TE_H1wr', 'FE', 'FEr', 'PE', 'PEn', 'PGE', 'PGEn',]
# QQ which to use?  >> let choose shorter set
# same plot with analytic value
plot2 = {'PG': 'APG'}  # NT: iplement ad hoc, does nothing
# QQ other plots?
create_timelines_plots()


# reports over seconds - generate plots =========================================================================

# QQ what to plot? Only CE_H1r or some merged characteristic to see if it is stable through cycles?

# for type_key in data_types.iterkeys():
#     print(type_key, data_types[type_key])
#     max_value = 0
#     empty_plot = True
#     colors = plt.get_cmap(color_set)(np.linspace(0, 1.0, data_types[type_key][0]))
#     i = 0
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
#         savefig('plots/S_' + type_key + '.png', lgd)
#     plt.close()

