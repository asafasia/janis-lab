# Copyright 2014-2021 Keysight Technologies
# -*- coding: utf-8 -*-
# for py2/py3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

def exportData(file_name, step_data, log_data, step_name, log_name, step_unit, log_unit, comment=''):
    """Export data from Log Viewer to file using custom format."""
    with open(file_name, 'w') as f:
        # write info
        f.write('Step channels:\n')
        for (name, unit) in zip(step_name, step_unit):
            f.write('%s [%s]\n' % (name, unit))
        f.write('\nLog channels:\n')
        for (name, unit) in zip(log_name, log_unit):
            f.write('%s [%s]\n' % (name, unit))
        f.write('\nComment:\n')
        f.write(comment + '\n')
        # 
        f.write('\nNumber of entries: %d\n' % len(step_data[0]))
        #
        # write data entries, x: first step channel, y: first log channel
        f.write('X-data:\n')
        for values in step_data[0]:
            sData = ', '.join([str(value) for value in values])
            f.write(sData + '\n')
        f.write('\n\nY-data:\n')
        for values in log_data[0]:
            sData = ', '.join([str(value) for value in values])
            f.write(sData + '\n')
        
        