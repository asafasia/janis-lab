# Copyright 2014-2021 Keysight Technologies
import numpy as np


def define_optimizer_settings():
    """Define extra settings for optimizer

    Returns
    -------
    optimizer_cfg : list of dict
        List of configuration items for optimizer, each item is a dict.
        Necessary keys are "name" and "datatype".

    """
    # Bayesian optimization settings
    optimizer_cfg = [
        dict(name='Acquisition function',
             datatype='COMBO',
             combo_defs=['LCB', 'EI', 'PI', 'gp_hedge'],
             def_value='gp_hedge',
             tooltip=('See https://scikit-optimize.github.io/ for more info'),
             ),
        dict(name='kappa',
             datatype='DOUBLE',
             def_value=1.96,
             state_item='Acquisition function',
             state_values=['LCB', 'gp_hedge'],
             tooltip=('Controls how much of the variance in the predicted ' +
                      'values should be taken into account. Higher value ' +
                      'favours exploration over exploitation and vice versa'),
             ),
        dict(name='xi',
             datatype='DOUBLE',
             def_value=0.1,
             state_item='Acquisition function',
             state_values=['EI', 'PI', 'gp_hedge'],
             tooltip=('Controls how much improvement one wants over the ' +
                      'previous best values. Higher value ' +
                      'favours exploration over exploitation and vice versa'),
             ),
    ]
    return optimizer_cfg


def optimize(config, minimize_function):
    """Optimize using scikit-optimize's gp_minimize

    Parameters
    ----------
    config : dict
        Optimizer settings provided from the measurement configuration

    minimize_function : callable
        Function to be minimized,
            evaluation_function(x) -> float
        The function will run Labber measurement for parameter values x.
        The function is typically passed to the scipy optimizer.

    Returns
    -------
    optimizer_result : dict
        Results from optimizer, using scipy's OptimizeResult format.
        Necessary keys are "x", containing the final optimizer parameters.

    """

    # import scikit-optimize here, to avoid errors when checking file format
    from skopt import gp_minimize

    # extract parameters
    x0 = [d['Start value'] for d in config['optimizer_channels']]

    # bounds
    dimensions = [(d['Min value'], d['Max value'])
                  for d in config['optimizer_channels']]

    # run optimizer for one less call than asked for, to catch final value
    n_calls = int(config['Max evaluations'] - 1)

    # optimize
    res = gp_minimize(
        minimize_function,
        dimensions=dimensions,
        n_calls=n_calls,
        x0=x0,
        acq_func=config['Acquisition function'],
        xi=config['xi'],
        kappa=config['kappa'],
        # acq_optimizer='sampling'
    )

    # only return relevant info
    res_out = dict(
        x=res['x'],
        success=res.get('success', True),
    )
    return res_out



if __name__ == '__main__':
    # test case for optimizer

    # define function for testing optimizer
    def func(x):
        a0 = 1.3
        b0 = 4.5
        return np.sqrt((x[0] - a0) ** 2 + (x[1] - b0) ** 2)

    # define configuration
    config = {
        'Minimization function': 'y[0]',
        'Method': 'Nelder-Mead',
        'Max evaluations': 20,
        'Relative tolerance': np.inf,
        'Target value': -np.inf,
        'Acquisition function': 'gp_hedge',
        'xi': 0.01,
        'kappa': 0.5,
        'optimizer_channels': [
            {'Enabled': True, 'channel_name': 'X', 'Start value': 9.8e-07,
             'Max value': 10.0, 'Min value': -10.0, 'Precision': .01,
             'Initial step size': 0.1},
            {'Enabled': True, 'channel_name': 'Y', 'Start value': 0.0,
             'Max value': 10.0, 'Min value': -10.0, 'Precision': .01,
             'Initial step size': 0.1},
        ]
    }
    print('Config:')
    print(config)

    # run test
    res = optimize(config, func)

    # print result
    print('Results:')
    print(res)
