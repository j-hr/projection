import csv
from dolfin.cpp.common import toc, tic, info


class TimeControl:
    def __init__(self):
        info('Initializing Time control')
        # watch is list [total_time, last_start, message_when_measured, count into total time]
        self.watches = {}
        self.last_measurement = 0
        self.measuring = 0
        tic()

    def init_watch(self, what, message, count_to_sum, count_to_percent=False):
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
            if from_last > 0.1:
                info('TC (%s): time from last end of measurement: %f' % (what, from_last))

    def end(self, what):
        watch = self.watches[what]
        elapsed = toc() - watch[1]
        watch[0] += elapsed
        if self.watches[what][3]:
            self.measuring -= 1
        info(watch[2]+'. Time: %.4f Total: %.4f' % (elapsed, watch[0]))
        self.last_measurement = toc()

    def report(self, report_file, str_name):
        total_time = toc()
        info('Total time of %.0f s, (%.2f hours).' % (total_time, total_time/3600.0))
        sorted_by_time = []
        sorted_by_name = []
        sum = 0
        sum_percent = 0
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
                    info('   %-40s: %12.2f s %5.1f %% (%4.1f %%)' % (value[2], value[0], 100.0*value[0]/sum_percent, 100.0*value[0]/total_time))
                else:
                    info('   %-40s: %12.2f s         (%4.1f %%)' % (value[2], value[0], 100.0*value[0]/total_time))
            else:
                info('   %-40s: %12.2f s NOT USED' % (value[2], value[0]))
        info('   %-40s: %12.2f s         (%4.1f %%)' % ('Measured', sum, 100.0*sum/total_time))
        info('   %-40s: %12.2f s 100.0 %% (%4.1f %%)' % ('Base for percent values', sum_percent, 100.0*sum_percent/total_time))
        info('   %-40s: %12.2f s %5.1f %%' % ('Unmeasured', total_time-sum, 100.0*(total_time-sum)/total_time))
        # report to file
        for key in self.watches.iterkeys():   # sort keys by name
            if not sorted_by_name:
                sorted_by_name.append(key)
            else:
                l = len(sorted_by_name)
                i = l
                while key < sorted_by_name[i-1] and i > 0:
                    i -= 1
                sorted_by_name.insert(i, key)
        report_header = ['Name', 'Total time']
        report_data = [str_name, total_time]
        for key in sorted_by_name:
            value = self.watches[key]
            report_header.append(value[2])
            report_header.append('part '+value[2])
            report_data.append(value[0])
            report_data.append(value[0]/total_time)
        report_header.append('part unmeasured')
        report_data.append((total_time-sum)/total_time)
        if report_file is not None:
            writer = csv.writer(report_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_NONE)
            writer.writerow(report_header)
            writer.writerow(report_data)

    def report_print(self):
        self.report(None, '')

