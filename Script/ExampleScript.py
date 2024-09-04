# Copyright 2014-2021 Keysight Technologies
# for py2/py3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np

from Labber import ScriptTools

# define list of points
vFlux = np.linspace(-1E-3, 1E-3, 101)

# define measurement objects
sPath = os.path.dirname(os.path.abspath(__file__))
MeasResonator = ScriptTools.MeasurementObject(os.path.join(sPath, 'TestResonator.hdf5'),
                                              os.path.join(sPath, 'TestResonatorOut.hdf5'))
MeasQubit = ScriptTools.MeasurementObject(os.path.join(sPath, 'TestQubit.hdf5'),
                                          os.path.join(sPath, 'TestQubitOut.hdf5'))
# set the primary channel that defines the third data dimension  
MeasResonator.setPrimaryChannel('Flux bias')
MeasQubit.setPrimaryChannel('Flux bias')

# go through list of points
for n1, value_1 in enumerate(vFlux):
    print('Flux [mA]:', 1000*value_1)
    # set flux bias
    MeasResonator.updateValue('Flux bias', value_1)
    MeasQubit.updateValue('Flux bias', value_1)
    # measure resonator
    (x,y) = MeasResonator.performMeasurement()
    # for this example, y is complex, take absolute value
    y = abs(y)
    # look for peak position
    print('Resonator position [GHz]:', x[np.argmax(y)]/1E9)
    # set new frequency position 
    MeasQubit.updateValue('RF - Frequency', x[np.argmax(y)])
    # measure qubit
    (x,y) = MeasQubit.performMeasurement()

