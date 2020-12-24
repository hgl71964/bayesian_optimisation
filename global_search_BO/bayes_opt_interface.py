import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy
import torch as tr

# import scripts 
from ..src.bayes_opt import bayesian_optimiser
from ..src.api_helper import api_utils

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
dtype = tr.float32

"""
import this script as a package

provide interface like this:

bayes_opt(loss, size, search_bounds, logger, init_queries, iteration)
"""
# hyperparameters
T = 30  # total number of iterations

# gp; includes "MA2.5", "SE", "RQ", "LR", "PO"
gp_name, gp_params = "MA2.5", {
                            "mode": "raw",      # "raw", "add", "pro" for GPs
                            "opt":"ADAM",  # opt for MLE; (quasi_newton, ADAM)
                            "epochs":128,       # epoch to run, if chosen ADAM
                            "lr":1e-1,          # learning rate for ADAM
                            }
# q-parallelism (if use analytical acq_func, q must be 1)
batch_size = 2

acq_params = { 
    "acq_name" : "qEI",          # acqu func; includes: "EI", "UCB", "qEI", "qUCB", "qKG"
    "N_start": 32,               # number of starts for multi-start SGA
    "raw_samples" :512,          # heuristic initialisation 
    "N_MC_sample" : 256,         # number of samples for Monte Carlo simulation
    "num_fantasies": 128,        # number of fantasies used by KG
    "beta":1.,                   # used by UCB/qUCB
            }



"""hyperparameter for bayes_opt"""

def bayes_loop(loss: callable,
            size: int,
            search_bounds: np.ndarray,
            logger,  # TODO change this 
            init_queries: np.ndarray,
            iteration: int, 
            ):

    bayes_opt = bayesian_optimiser(gp_name, gp_params, device, acq_params)

    # TODO

    xs, ys = bayes_opt.outer_loop(T, (-4,4), x0, y0, r0, api, batch_size)  # maximising reward






