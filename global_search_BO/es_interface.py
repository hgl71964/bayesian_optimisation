import numpy as np
import torch as tr

# import scripts 
from .src.es import evolutionary_strategy
from .src.api_helper import api_utils
dtype = tr.float32

"""
import this script as a package

provide interface like this:

evolutionary_strategy(loss, iteration, size, search_bounds, logger, 
                                es_params, device)
"""

def es_loop(loss_func: callable,
            iteration: int,  # time horison
            size: int,  # q-parallelism, this should be 1 in ES
            search_bounds: np.ndarray,  # shape: ((2, d)) 
            logger,  # TODO change this 
            es_params: dict, 
            device = tr.device("cpu"),  # change to gpu if possile 
            ):
    es = evolutionary_strategy(size, **es_params)

    # get x0, y0
    x0 = tr.from_numpy(api_utils.init_query(size, search_bounds)).float().to(device)

    #  decorate the api
    api = api_utils.wrapper(loss_func); r0 = 0; # TODO think about normalisation
                                                # TODO add search_bound restriction
    return es.outer_loop(iteration, x0, r0, api)
