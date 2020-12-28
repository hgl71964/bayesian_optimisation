import numpy as np
from copy import deepcopy
import torch as tr

# import scripts 
from .src.bayes_opt import bayesian_optimiser
from .src.api_helper import api_utils
dtype = tr.float32

"""
import this script as a package

provide interface like this:

bayes_opt(loss, iteration, size, search_bounds, logger, device)
"""

def bayes_loop(loss_func: callable,
            iteration: int,  # time horison
            size: int,  # q-parallelism (if use analytical acq_func, q must be 1)
            search_bounds: np.ndarray,  # shape: ((2, d))
            logger,  # TODO change this 
            gp_name,
            gp_params,
            acq_params,
            device = tr.device("cpu"),  # change to gpu if possile 
            ):
    bayes_opt = bayesian_optimiser(gp_name, gp_params, device, acq_params)

    # get x0, y0
    x0 = api_utils.init_query(size, search_bounds); y0 = api_utils.init_reward(x0, loss_func);

    #  format the initial pair
    x0, y0 = tr.from_numpy(x0).float().to(device), y0.to(device)

    #  decorate the api
    api = api_utils.wrapper(loss_func); r0 = 0; # TODO think about normalisation

    return bayes_opt.outer_loop(iteration, search_bounds, x0, y0, r0, api, size)
