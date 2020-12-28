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

bayes_opt(loss, size, search_bounds, logger, init_queries, iteration)
"""

# gp; includes "MA2.5", "SE", "RQ", "LR", "PO"
gp_name, gp_params = "MA2.5", {
                            "mode": "raw",      # "raw", "add", "pro" for GPs
                            "opt":"ADAM",  # opt for MLE; (quasi_newton, ADAM)
                            "epochs":128,       # epoch to run, if chosen ADAM
                            "lr":1e-1,          # learning rate for ADAM
                            }

acq_params = { 
    "acq_name" : "qEI",          # acqu func; includes: "EI", "UCB", "qEI", "qUCB", "qKG"
    "N_start": 32,               # number of starts for multi-start SGA
    "raw_samples" :512,          # heuristic initialisation 
    "N_MC_sample" : 256,         # number of samples for Monte Carlo simulation
    "num_fantasies": 128,        # number of fantasies used by KG
    "beta":1.,                   # used by UCB/qUCB
            }

"""hyperparameter for bayes_opt"""

def bayes_loop(loss_func: callable,
            iteration: int,  # time horison
            size: int,  # q-parallelism (if use analytical acq_func, q must be 1)
            search_bounds: np.ndarray,  # shape: ((n, 2))
            logger,  # TODO change this 
            device = tr.device("cpu"),  # change to gpu if possile 
            ):
    """format hyper-parameters"""
    global  gp_name, gp_params, acq_params  # TODO: change this 
    """end of format hyper-parameters"""

    bayes_opt = bayesian_optimiser(gp_name, gp_params, device, acq_params)

    # get x0, y0
    x0 = api_utils.init_query(size, search_bounds); y0 = api_utils.init_reward(x0, loss_func);

    #  format the initial pair
    x0, y0 = tr.from_numpy(x0).float().to(device), y0.to(device)

    #  decorate the api
    api = api_utils.wrapper(loss_func); r0 = 0; # TODO think about normalisation

    return bayes_opt.outer_loop(iteration, search_bounds, x0, y0, r0, api, size)
