# Copyright 2014-2021 Keysight Technologies
# This file is part of the Labber API, a Python interface to Labber.
#
# http://keysight.com/find/labber
#
# All rights reserved

"""
This is a Python interface to Labber.
Software for Instrument Control and Lab Automation.
"""

from __future__ import absolute_import as _ai
from sys import version_info as _info

import os
os.environ['QT_API'] = 'pyqt5'

# get import folder depending python version
_VERSION_ERROR_STRING = 'The Labber API requires Python >=3.6 and <=3.9'

if _info >= (3, 10) or _info < (3, 6):
    raise ImportError(_VERSION_ERROR_STRING)

elif _info >= (3, 9):
    # version info
    from ._include39._version import version as __version__
    from ._include39._version import info as version
    # script tools
    from ._include39 import _ScriptTools as ScriptTools
    # log file
    from ._include39._LogFile import (
        LogFile, createLogFile_ForData, getTraceDict)
    # labber client
    from ._include39._Client import connectToServer
    # scripting with scenarios
    from ._include39 import _config as config
    from ._include39.labber.config.scenario import Scenario
    # requires PyQt, ignore if not present
    # try:
    #     from ._include39.LabberClient import (
    #         LabberClient, LabberBlockingClient, InstrumentClient)
    # except Exception:
    #     pass

elif _info >= (3, 8):
    # version info
    from ._include38._version import version as __version__
    from ._include38._version import info as version
    # script tools
    from ._include38 import _ScriptTools as ScriptTools
    # log file
    from ._include38._LogFile import (
        LogFile, createLogFile_ForData, getTraceDict)
    # labber client
    from ._include38._Client import connectToServer
    # scripting with scenarios
    from ._include38 import _config as config
    from ._include38.labber.config.scenario import Scenario
    # requires PyQt, ignore if not present
    # try:
    #     from ._include38.LabberClient import (
    #         LabberClient, LabberBlockingClient, InstrumentClient)
    # except Exception:
    #     pass

elif _info >= (3, 7):
    # version info
    from ._include37._version import version as __version__
    from ._include37._version import info as version
    # script tools
    from ._include37 import _ScriptTools as ScriptTools
    # log file
    from ._include37._LogFile import (
        LogFile, createLogFile_ForData, getTraceDict)
    # labber client
    from ._include37._Client import connectToServer
    # scripting with scenarios
    from ._include37 import _config as config
    from ._include37.labber.config.scenario import Scenario
    # requires PyQt, ignore if not present
    # try:
    #     from ._include37.LabberClient import (
    #         LabberClient, LabberBlockingClient, InstrumentClient)
    # except Exception:
    #     pass

elif _info >= (3, 6):
    # version info
    from ._include36._version import version as __version__
    from ._include36._version import info as version
    # script tools
    from ._include36 import _ScriptTools as ScriptTools
    # log file
    from ._include36._LogFile import (
        LogFile, createLogFile_ForData, getTraceDict)
    # labber client
    from ._include36._Client import connectToServer
    # scripting with scenarios
    from ._include36 import _config as config
    from ._include36.labber.config.scenario import Scenario
    # # requires PyQt, ignore if not present
    # try:
    #     from ._include36.LabberClient import (
    #         LabberClient, LabberBlockingClient, InstrumentClient)
    # except Exception:
    #     pass

else:
    raise ImportError(_VERSION_ERROR_STRING)
