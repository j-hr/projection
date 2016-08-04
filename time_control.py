import csv
from dolfin.cpp.common import toc, tic, info


class TimeControl:
    """
    Used to measure and report time spent in different steps of program.
    Time for program run can be distorted by JIT compilation. Make sure all code was compiled before taking time
    measurements seriously.

    Time control writes messages starting 'TC' in two cases
        1. if you count time more than once
        2. if there is more than 0.1 s time interval that was not measured at all (to easily find where the unmeasured
        time is)
    """
    def __init__(self):
        info('Initializing Time control')
        # watch is list [total_time, last_start, message_when_measured, count into total time, count into selected time]
        self.watches = {}
        self.last_measurement = 0
        self.measuring = 0
        tic()

    def init_watch(self, what, message, count_to_sum, count_to_percent=False):
        """
        For every task to be measured, watch must be set up.
        :param what: string code to identify watch
        :param message: written out when time is added to watch (after the measured process)
        :param count_to_sum: if watches overlap, set to False so time is not counted twice into total
        :param count_to_percent: this way you can select subset of watches to see their ratio without interference of
        other watches (e. g. if you want to compare time in solvers without time used to save files)
        """
        if what not in self.watches:
            self.watches[what] = [0, 0, message, count_to_sum, count_to_percent]

    def start(self, what):
        if what in self.watches:
            if self.watches[what][3]:
                self.measuring += 1
            self.watches[what][1] = toc()
            from_last = toc()-self.last_measurement
            if self.measuring > 1:
                info('TC (%s): More watches at same time: %d' % (what, self.measuring))
            elif from_last > 0.1 and self.watches[what][3]:
                info('TC (%s): time from last end of measurement: %f' % (what, from_last))

    def end(self, what):
        watch = self.watches[what]
        elapsed = toc() - watch[1]
        watch[0] += elapsed
        if self.watches[what][3]:
            self.measuring -= 1
            self.last_measurement = toc()
        info(watch[2]+'. Time: %.4f Total: %.4f' % (elapsed, watch[0]))

    def report(self, report_file, str_name):
        """
        Writes out complete report. Saves times for the selected watches to csv file (if given).
        """
        total_time = toc()
        info('Total time of %.0f s, (%.2f hours).' % (total_time, total_time/3600.0))
        sorted_by_time = []
        sorted_by_name = []
        sum = 0
        sum_percent = 0
        # sort watches by time measured
        for value in self.watches.itervalues():
            if value[3]:
               sum += value[0]
            if value[4]:
               sum_percent += value[0]
            if not sorted_by_time:
                sorted_by_time.append(value)
            else:
                i = 0
                l = len(sorted_by_time)
                while i < l and value[0]<sorted_by_time[i][0]:
                    i += 1
                sorted_by_time.insert(i, value)
        for value in sorted_by_time:
            if value[0] > 0.000001:
                if value[4]:
                    info('   %-40s: %12.2f s %5.1f %% (%4.1f %%)' % (value[2], value[0], 100.0*value[0]/sum_percent,
                                                                     100.0*value[0]/total_time))
                else:
                    info('   %-40s: %12.2f s         (%4.1f %%)' % (value[2], value[0], 100.0*value[0]/total_time))
            else:
                info('   %-40s: %12.2f s NOT USED' % (value[2], value[0]))
        info('   %-40s: %12.2f s         (%4.1f %%)' % ('Measured', sum, 100.0*sum/total_time))
        info('   %-40s: %12.2f s 100.0 %% (%4.1f %%)' % ('Base for percent values', sum_percent,
                                                         100.0*sum_percent/total_time))
        info('   %-40s: %12.2f s         (%4.1f %%)' % ('Unmeasured', total_time-sum,
                                                        100.0*(total_time-sum)/total_time))
        # report to file
        report_header = ['Name', 'Total time']
        report_data = [str_name, total_time]
        for key in sorted(self.watches.keys()):
            value = self.watches[key]
            if value[4]:
                report_header.append(value[2])
                report_data.append(value[0])
        if report_file is not None:
            writer = csv.writer(report_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            writer.writerow(report_header)
            writer.writerow(report_data)

    def report_print(self):
        self.report(None, '')

