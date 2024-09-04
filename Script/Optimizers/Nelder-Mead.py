# Copyright 2014-2021 Keysight Technologies
from scipy.optimize import minimize
import numpy as np


def optimize(config, minimize_function):
    """Optimize using Scipy's built-in Nelder-mead optimizer

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
    # extract parameters
    x0 = np.array([d['Start value'] for d in config['optimizer_channels']])

    # nelder-mead in scipy does not support bounds, not handled here
    # bounds = [(d['Min value'], d['Max value'])
    #           for d in config['optimizer_channels']]

    x_tolerance = [d['Precision'] for d in config['optimizer_channels']]
    # nelder-mead in scipy only supports single value for x tolerance, but
    # parameters are re-scaled by Labber if necessary, so all values are equal
    xatol = min(x_tolerance)

    # define initial simplex from step size parameters
    step_size = [d['Initial step size'] for d in config['optimizer_channels']]
    initial_simplex = np.zeros((len(x0) + 1, len(x0)))
    # first point is start value
    initial_simplex[0] = x0
    # other points are defined by initial step size
    for n, channel in enumerate(config['optimizer_channels']):
        initial_simplex[n + 1] = x0
        # go in direction furthest from bound
        if (x0[n] - channel['Min value']) <= (channel['Max value'] - x0[n]):
            # closer to min value, go positive
            initial_simplex[n + 1, n] += step_size[n]
        else:
            # closer to max value, go negative
            initial_simplex[n + 1, n] -= step_size[n]

    # creat options for minimizer
    options = dict(
        maxiter=config['Max evaluations'],
        maxfev=config['Max evaluations'],
        fatol=config['Relative tolerance'],
        xatol=xatol,
        initial_simplex=initial_simplex,
    )

    # optimize
    res = minimize(
        minimize_function, x0,
        method='Nelder-Mead',
        options=options,
    )

    return res



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
        'Max evaluations': 500,
        'Relative tolerance': np.inf,
        'Target value': -np.inf,
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
