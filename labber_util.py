# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:00:15 2022

@author: Guy
"""
import pprint

import Labber as lb
import os
import numpy as np


def get_current_Labber_path():
    """
    Labber has a database structure based on dates and it automatically writes new logfiles into the folder of today/
    however here we want to be absolutely sure that we get the correct folder so we find it by creating a fictitious
    temporary  logfile called temp.  and get its path. not very elegant but it works.

    :return:str: path to the current Labber database folder
    """
    # create a temporary log:
    lLog = [dict(name='name', vector=False)]
    templog = lb.createLogFile_ForData('temp', lLog)
    # get its path
    templog_path = templog.getFilePath(None)

    # return just the folder
    return os.path.dirname(templog_path)


def get_logfile_full_path(log_name):
    """
    returns the full path of a Labber logfile with name log_name in the current Labber database folder. note that the
    logfile might be a nonexisting one and that's ok. the function will just return a full path pf the form f'{Labber
    folder}/{log_name}.hdf5' where labber folder is the current Labber database folder, and that can be used to create
    a new logfile

    :param log_name: str
    :return: str: full path: '{Labber folder}/{log_name}.hdf5'
    """
    log_name = f'{log_name}.hdf5'
    return os.path.join(get_current_Labber_path(), log_name)


def log_exists(log_full_path):
    """
    checks whether a Labber logfile with name full path log_path_name exists in the current Labber database directory.
    :param log_name:str
    :return: bool
    """
    return os.path.exists(log_full_path)


def get_log_name(log_name):
    """
    if log_name does not exist in the current Labber database directory, returns log_name
    if it does exist, adds a number at the end as needed to avoid overwrite.
    example: if the current Labber folder has the following files:
    my_experiment.hdf5
    my_experiment__1.hdf5
    my_experiment__2.hdf5
    my_experiment__3.hdf5

    then  get_log_name('my_experiment') = 'my_experiment__4'
    :param log_name:str desired name for labber logfile
    :return:str the same name with additional numbers at the end as necessary to avoid overwrite.
    """
    labber_path = get_current_Labber_path()
    LOG_NAME_FORMAT = '{log_name}__{integer_suffix}'
    counter = 1
    log_name_temp = log_name

    while log_exists(os.path.join(labber_path, f'{log_name_temp}.hdf5')):
        log_name_temp = LOG_NAME_FORMAT.format(log_name=log_name, integer_suffix=counter)
        counter = counter + 1
    return log_name_temp


def open_log(log_name):
    """
    returns a Labber.logfile object corresponding to the data file indicated by log_name from the current Labber
    database folder
    :param log_name: a name of a log file that exists in the current database
    :return:Labber.logfile object
    """
    return lb.LogFile(get_logfile_full_path(log_name))


def create_logfile(name, loop_type="1d", units=None, **kwargs, ):
    """
    create a new Labber log file from experiment data

    Args:
        name: str  - the name of the logfile, to which will be added a number if needed to avoid overwrite

        loop_type: {'1d' | '2d'}  other loop dimensions are currently not supported

        units (optional):  dict with keys that correspond to some sweep parameter names and/or measured parameters names
        (not necessary for all of them), and values that are strings. example: {'time':'s', 'frequency':'Hz',
        "signal" : 'a.u.'}. usually it is better not to specify a unit than to use something like 'a.u.'

    required keyword arguments:

        sweep_parameters: dict in the format {name1 : values1, name2 : values2} and so on.
        values should be one-dimensional iterable. example: {'time': [0.1,0.2,0.3], 'frequency': [100, 200, 300, 400]}

        measured_data: dict in the format {name1 : values1, name2 : values2} and so on. values should be an np-array
        with all the measured data for this channel. complex data is also supported

        meta_data (optional): some dict with metadata that will be printed into the labber comment.
        optionally, it can have a "tags" key with value that is a list of strings corresponding to required labber tags.
        similarly it can optionally have a "user" key with a string value that will be added as the labber user.



    Returns: a Labber.LogFile object corresponding to the created log file

    """
    exp_result = kwargs
    # get new name and prevent overwrite:
    logfile_name = get_log_name(name)
    # create step list
    step_names = exp_result["sweep_parameters"].keys()
    lStep = []
    for step in step_names:
        if units is not None and step in units.keys():
            lStep.append(dict(name=step, unit=units[step], values=exp_result["sweep_parameters"][step]))
        else:
            lStep.append(dict(name=step, values=exp_result["sweep_parameters"][step]))

    # create log channels list
    log_names = exp_result["measured_data"].keys()
    lLog = []
    for log in log_names:
        complex_flag = np.iscomplexobj(exp_result["measured_data"][log])
        if units is not None and log in units.keys():
            lLog.append(
                dict(name=log, unit=units[log], vector=False, complex=complex_flag))  # TODO: support vector channels
        else:
            lLog.append(dict(name=log, vector=False, complex=complex_flag))  # TODO: support vector channels

    # create labber logfile
    lf = lb.createLogFile_ForData(logfile_name, lLog, lStep)
    # insert data to log channels
    if loop_type == '2d':
        for outer_loop_index, outer_step_value in enumerate(lStep[1]["values"]):
            labber_dict = {}
            for log in log_names:
                labber_dict[log] = (exp_result["measured_data"][log][outer_loop_index])
            lf.addEntry(labber_dict)
    elif loop_type == '1d':
        labber_dict = {}
        for log in log_names:
            labber_dict[log] = (exp_result["measured_data"][log])
        lf.addEntry(labber_dict)
    else:
        raise ValueError("loop_type must be '2d' or '1d'.")

    # add metadata
    lf.setComment(pprint.pformat(exp_result["meta_data"]))
    if "tags" in exp_result["meta_data"].keys():
        lf.setTags(exp_result["meta_data"]["tags"])

    if "user" in exp_result["meta_data"].keys():
        lf.setUser(exp_result["meta_data"]["user"])

    return lf
