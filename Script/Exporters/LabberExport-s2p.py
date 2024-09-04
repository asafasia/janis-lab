# Copyright 2014-2021 Keysight Technologies
# -*- coding: utf-8 -*-
# for py2/py3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

def exportData(file_name, step_data, log_data, step_name, log_name, step_unit, log_unit, comment=''):
    """Export data from Log Viewer to s2p format."""
    # only supports 1-port parameter files
    with open(file_name, 'w') as f:
        # get data
        vFreq = step_data[0][0]/1E9
        vS = log_data[0][0]
        # check if scalar or vector data
        if len(vS) != len(vFreq):
            # frequency defined by vector data
            vFreq = log_data[1][0]/1E9
        # write info
        f.write('!1-port S-parameter file, multiple frequency points\n')
        f.write('[Version] 2.0\n')
        f.write('# GHz S RI\n')
        f.write('[Number of Ports] 1\n')
        f.write('[Number of Frequencies] %d\n' % len(vFreq))
        f.write('[Reference] 50.0\n')
        f.write('[Network Data]\n')
        f.write('!freq reS11 imS11\n')
        # write values
        for freq, value in zip(vFreq, vS):
            f.write('%.9f\t%.9f\t%.9f\n' % (freq, value.real, value.imag))
        
        