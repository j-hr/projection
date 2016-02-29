from __future__ import print_function
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import problem as prb
import math
import numpy
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
factors = range(1, 6)
f_str = {1: '0.01', 2: '0.05', 3: '0.1', 4: '0.5', 5: '1.0'}
meshes = range(1, 4)
dts = range(1, 6)
dtToMs = {1: 100, 2: 50, 3: 10, 4: 5, 5: 1}
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
            # print(times[i])
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
    formats3 = [['x-.', '+--', '1-'],
                ['+-.', 'x--', '2-']]
    line_widths3 = [2.0, 1.5, 1.0]
    formats5 = ['1-', '2-', '3-', 'x-', '+-']
    line_width5 = 1.0
    marker_size = 10.0
    marker_edge_width = 2.0
    for (plot_name, plot) in c_plots.iteritems():
        print(plot_name)
        figT = plt.figure(1)
        figS = plt.figure(2)
        splT = figT.add_subplot('111')
        splS = figS.add_subplot('111')
        plot_empty = True
        max_ = -1e16
        min_ = 1e16
        for ch in plot['characteristics']:
            max_plotted_value = 10 if str(ch).endswith('r') else 1e10
            for prb in plot['problems']:
                indices = {'problems': prb, 'characteristics': ch}
                for f in plot['factors']:
                    for m in meshes:
                        Tvalues = []
                        Tx = []
                        for t in dts:
                            name = prb + ('%d%d%d' % (f, m, t))
                            print('  Looking for:', name)
                            if name in problems[prb]:
                                print('    Found!')
                                d = problems[prb][name]
                                value = d['report'][ch_index[ch]]
                                if 0 < value < max_plotted_value:  # do not plot values 0 and from diverging problems
                                    Tvalues.append(value)
                                    Tx.append(dtToMs[t])
                                    min_, max_ = minmax(min_, max_, value, value)
                        if Tvalues:
                            plot_empty = False
                            idx = plot[plot['colors']].index(indices[plot['colors']])
                            rng = len(plot[plot['colors']])
                            splT.plot(Tx, Tvalues, formats3[idx % 2][m-1], lw=line_widths3[m-1],
                                      label=plot['label'](prb, f, ch, m, t) + ' on mesh %d' % m, color=color(idx, rng),
                                      ms=marker_size, mew=marker_edge_width)
                    for t in dts:
                        Svalues = []
                        Sx = []
                        for m in meshes:
                            name = prb + ('%d%d%d' % (f, m, t))
                            print('  Looking for:', name)
                            if name in problems[prb]:
                                print('    Found!')
                                d = problems[prb][name]
                                value = d['report'][ch_index[ch]]
                                if 0 < value < max_plotted_value:  # do not plot values 0 and from diverging problems
                                    Svalues.append(value)
                                    Sx.append(d['md']['h'])
                        if Svalues:
                            idx = plot[plot['colors']].index(indices[plot['colors']])
                            rng = len(plot[plot['colors']])
                            splS.plot(Sx, Svalues, formats5[t-1], lw=line_width5,
                                      label=plot['label'](prb, f, ch, m, t) + ' with dt=%d' % dtToMs[t],
                                      color=color(idx, rng), ms=marker_size, mew=marker_edge_width)
        if not plot_empty:
            print('  Plotting')
            for a in [figT.axes[0], figS.axes[0]]:
                a.set_title(plot_name)
                a.set_xscale('log')
                a.set_yscale('log')
                a.set_ylim(min_*0.95, max_*1.05)
                # TODO add ticks and ticklabels for min and max values
                #a.set_yticks(a.get_yticks().tolist() + [min_, max_])
                #a.set_yticklabels(a.get_yticklabels().extend([min_, max_]))
            axesT = figT.axes[0]
            splT.set_xlabel('dt in ms')
            splT.set_xlim(100.5, 0.95)
            lgdT = axesT.legend(bbox_to_anchor=(1.5, 1.0))
            savefig(figT, 'plots/C_' + plot_name + '_CT.png', lgdT)

            axesS = figS.axes[0]
            splS.set_xlabel('h')
            splS.set_xlim(2.3, 0.5)
            axesS.set_xticks([2.0, 1.0, 0.5])
            axesS.set_xticklabels(['2.0', '1.0', '0.5'])
            lgdS = axesS.legend(bbox_to_anchor=(1.5, 1.0))
            savefig(figS, 'plots/C_' + plot_name + '_CS.png', lgdS)

        plt.figure(1)
        plt.close()
        plt.figure(2)
        plt.close()


def create_timelines_plots():
    formats = {2: ['--', '-'],
               3: ['-.', '--', '-'],
               4: [':', '-.', '--', '-']
               }
    line_widths = {2: [1.5, 1.0],
                   3: [2.0, 1.5, 1.0],
                   4: [2.5, 2.0, 1.5, 1.0]
                   }
    for (plot_name, plot) in t_plots.iteritems():
        print(plot_name)
        fig = plt.figure(1)
        s1 = fig.add_subplot('121')
        s2 = fig.add_subplot('122')
        plot_empty = True
        max_ = -1e16
        min_ = 1e16
        max_l = -1e16
        min_l = 1e16
        for ch in plot['characteristics']:
            for prb in plot['problems']:
                for f in plot['factors']:
                    for t in plot['times']:
                        for m in plot['meshes']:
                            indices = {'meshes': m-1, 'times': t-1, 'problems': plot['problems'].index(prb),
                                       'characteristics': plot['characteristics'].index(ch)}
                            name = prb + ('%d%d%d' % (f, m, t))
                            print('  Looking for:', name)
                            if name in problems[prb] and ch in problems[prb][name]['timelines']:
                                print('    Found!')
                                d = problems[prb][name]
                                y = d['timelines'][ch]
                                x = times[t]
                                min_, max_ = minmax(min_, max_, min(y[dtToSteps[t]:]), max(y[dtToSteps[t]:]))
                                min_l, max_l = minmax(min_l, max_l, min(y[(d['md']['cycles']-1)*dtToSteps[t]:]), max(y[(d['md']['cycles']-1)*dtToSteps[t]:]))
                                plot_empty = False
                                s2.set_yscale('log')
                                for s in [s1, s2]:
                                    s.plot(x, y, formats[len(plot[plot['formats']])][indices[plot['formats']]],
                                           label=plot['label'](prb, f, ch, m, t),
                                           color=color(indices[plot['colors']], len(plot[plot['colors']])),
                                           lw=line_widths[len(plot[plot['formats']])][indices[plot['formats']]])
        if not plot_empty:
            print('  Plotting')
            for s in [s1, s2]:
                s.set_xlabel('time')
                s.set_xlim(0, d['md']['cycles'])
            min_log = math.pow(10, math.floor(math.log10(min_)))
            s1.set_ylim(min_, max_)
            s2.set_ylim(min_log, max_)
            s2.set_title(plot_name + ' for factor=' + f_str[f])
            lgd = plt.legend(bbox_to_anchor=(1.0 + plot['legend size'], 1.0))
            savefig(plt.gcf(), 'plots/TL_' + plot_name + '_f%d' % f + '.png', lgd)
            # save same plot only for last cycle
            for s in [s1, s2]:
                s.set_xlabel('time (last cycle)')
                s.set_xlim((d['md']['cycles']-1), d['md']['cycles'])
            min_log = math.pow(10, math.floor(math.log10(min_l)))
            s1.set_ylim(min_l, max_l)
            s2.set_ylim(min_log, max_l)
            savefig(plt.gcf(), 'plots/TL_' + plot_name + '_f%d' % f + 'lc.png', lgd)
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
# TODO compare 'PEn' amd 'PTEn' for rotation schemes
c_plots = {}
# conv. plot: more problems, one characteristic
compare_problems = {'all': problem_list,
          # 'normal vs no3bc': ['IBC_I', 'IBCbI'],
          # 'normal vs rotation': ['IBC_I', 'IBCRI'],
          # 'rotation vs rot+no3bc': ['IBCRI', 'IBCrI'],
          # 'all rotation': ['IBCRI', 'IBCrI', 'IBCRD', 'IBCRd'],
          # 'rotation Krylov vs direct': ['IBCRI', 'IBCRD'],
          # 'rotation Krylov vs direct + no3bc': ['IBCrI', 'IBCRd']}
}
# for oneplot in compare_problems['all']:
#     compare_problems[oneplot] = [oneplot]
# characteristics = ['time', 'CE_L2r', 'CE_H1r', 'CE_H1wr', 'PEn', 'FEr']
characteristics = ['time', 'CE_H1r', 'PEn', 'FEr']
# characteristics_single = ['time', 'PEn', 'FEr']
for (pr_name, prbs) in compare_problems.iteritems():
    for f in [1, 3]:
        for chs in characteristics:
            name = '%s %s f=%s' % (pr_name, chs, f_str[f])
            c_plots[name] = {
                'problems': prbs, 'characteristics': [chs],
                'colors': 'problems', 'factors': [f],
                'label': lambda prb, f, ch, m, t: '%s' % prb
            }
# conv. plot: one problem, more characteristics
compare_params = {'velocity error norms': ['CE_L2r', 'CE_H1r', 'CE_H1wr'],
                  'tentative vs corected error H1': ['TE_H1r', 'CE_H1r'],
                  #'tentative vs corected error H1 wall': ['TE_H1wr', 'CE_H1wr'],
                  }
# for char in characteristics_single:
#     compare_params[char] = [char]
for (ch_name, chs) in compare_params.iteritems():
    for f in [1]:
        for prb in problem_list:
            name = '%s %s f=%s' % (prb, ch_name, f_str[f])
            c_plots[name] = {
                'problems': [prb], 'characteristics': chs,
                'colors': 'characteristics', 'factors': [f],
                'label': lambda prb, f, ch, m, t: '%s' % ch
            }
create_convergence_plots()

# define plots:
# plot: one parameter, one problem, one factor, 3 meshes, 5 dts
t_plots = {}
# plot1 = ['CE_H1r', 'PEn']
plot1 = []
for problem in problem_list:
    for f in [1]:
        for char in plot1:
            name = problem + ' ' + char + ' f=' + f_str[f]
            t_plots[name] = {
                'characteristics': [char], 'problems': [problem],
                'factors': [f], 'times': dts, 'meshes': meshes,
                'colors': 'times', 'formats': 'meshes',
                'label': lambda prb, f, ch, m, t: '%dms on mesh %d' % (dtToMs[t], m),
                'legend size': 1.0
            }
# QQ which to use?  >> let choose shorter set
# same plot with analytic value
plot2 = {'PG': 'APG'}  # NT: iplement ad hoc, does nothing
# QQ other plots?
# plot: compare more parameters, one problem, one factor, 1 mesh, 5 dts
compare_params = {# 'velocity error norms': ['CE_L2r', 'CE_H1r', 'CE_H1wr'],
                  # 'tentative vs corected error H1': ['TE_H1r', 'CE_H1r'],
                  # 'tentative vs corected error H1 wall': ['TE_H1wr', 'CE_H1wr'],
                  'force error composition': ['FNE', 'FSE', 'FE'],
                  'force error relative composition': ['FNEr', 'FSEr', 'FEr'],
                  }
for problem in problem_list:
    for f in [1]:
        for m in meshes:
            for (plot_name, params) in compare_params.iteritems():
                name = problem + ' ' + plot_name + ' on mesh %d ' % m + 'f=' + f_str[f]
                t_plots[name] = {
                    'characteristics': params, 'problems': [problem],
                    'factors': [f], 'times': dts, 'meshes': [m],
                    'colors': 'times', 'formats': 'characteristics',
                    'label': lambda prb, f, ch, m, t: '%s %dms' % (ch, dtToMs[t]),
                    'legend size': 1.0
                }

# plot: compare more problems, one parameter, one factor, 1 mesh, 5 dts
compare_problems.pop('all')
compare_problems = {}
for ch in ['CE_H1r', 'PEn', 'FEr']:
    for f in [1]:
        for m in meshes:
            for (plot_name, prbs) in compare_problems.iteritems():
                name = plot_name + ' ' + ch + ' on mesh %d ' % m + 'f=' + f_str[f]
                t_plots[name] = {
                    'characteristics': [ch], 'problems': prbs,
                    'factors': [f], 'times': dts, 'meshes': [m],
                    'colors': 'times', 'formats': 'problems',
                    'label': lambda prb, f, ch, m, t: '%s %dms' % (prb, dtToMs[t]),
                    'legend size': 1.0
                }

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

