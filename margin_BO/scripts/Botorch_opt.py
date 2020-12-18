"""
2020 Summer internship

implement a botorch bayesian optimiser
"""

import torch
import numpy as np
import copy
from . import GPs  #  this script should be imported as packages
import time
import datetime
import botorch
import gpytorch


class bayesian_optimiser:
    """
    data type assume torch.tensor.float()

    the optimiser is set to MAXIMISE function!
    """
    def __init__(self, gp_name, gp_params, params):
        """
        Args:
            gp_nanme: name of the gp model; str
            gp_params: hyper-parameter for gp; dict
        kwargs:
            parameters for acquisiton function
        """
        self.params = params
        self.gpr = self._init_GPs(gp_name, gp_params)  #  instantiate GP

    def outer_loop(self, T, x, y, m0, api, batch_size):
        """
        standard bayesian optimisation loop

        Args:
            T: time_horizon;
            x: init samples; shape [n,d] -> n samples, d-dimensional
            y: shape shape [n,1]; 1-dimensional output
            m0: initial margin, on which the normalisation is based; float
            batch_size: q-parallelism; int
            api: callable; -> reward = api(query, m0)

        Returns:
            x,y: collection of queries and rewards; torch.tensor
        """
        # bounds for input dimension; assume dtype = torch.float; shape [2,d]
        input_dim = x.size(-1)
        bounds = torch.tensor([[0] * input_dim, [1] * input_dim], dtype = torch.float)

        mll, model = self.gpr.init_model(x, y, state_dict=None)
        times = [None] * T

        for t in range(T):

            # fit model every round
            self.gpr.fit_model(mll, model, x, y)

            #  timing
            acq_func_time = time.time()

            # acquisition function && query
            acq = self._init_acqu_func(model, y)
            query = self._inner_loop(acq, batch_size, bounds)

            #  timing 
            middle_time = time.time()

            # reward
            reward = api(query, m0)

            api_time = time.time()

            #  runtime report
            print(f"acq_func took {(middle_time - acq_func_time):.1f}s")
            print(f"api took {(api_time - middle_time):.1f}s")
            times[t] = middle_time - acq_func_time

            # append available data && update model
            x = torch.cat([x, query])
            y = torch.cat([y, reward])
            mll, model = self.gpr.init_model(x, y, state_dict=model.state_dict())

            print(f"time step {t+1}, drop {100*(1+reward.max()):,.2f}%; min ${-reward.max()*m0:,.0f}")
        
        print(f"acq_func average runtime per iteration {(sum(times)/len(times)):.1f}s")
        return x, y

    def _inner_loop(self, acq_func, batch_size, bounds):
        candidates, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=self.params["N_start"],       # number of starting point SGD
        raw_samples=self.params["raw_samples"],    # heuristic init
        sequential = False,                        # this enable SGD, instead of one-step optimal
        )
        query = candidates.detach()
        return query

    def _init_acqu_func(self,model,ys):
        if self.params["acq_name"] =="qKG":
            """
            if use sampler && resampling then slower
            """
            # sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.qKnowledgeGradient(
                model=model,
                num_fantasies=self.params["num_fantasies"],
                # sampler=sampler,
                objective=None,
            )
        elif self.params["acq_name"] == "qEI":
            sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
                model=model,
                best_f=ys.max(),
                sampler=sampler,
                objective=None, # identity objective; potentially useful model with constraints
            )
        elif self.params["acq_name"] == "qUCB":
            sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.monte_carlo.qUpperConfidenceBound(
                model=model,
                beta=self.params["beta"],
                sampler=sampler,
                objective=None,
            )
        elif self.params["acq_name"] == "EI":
            acq = botorch.acquisition.analytic.ExpectedImprovement(
                model=model,
                best_f=ys.max(),
                objective=None,
            )
        elif self.params["acq_name"] == "UCB":
            acq = botorch.acquisition.analytic.UpperConfidenceBound(
                model = model,
                beta = self.params["beta"],
                objective = None,
            )
        # elif self.params["acq_name"] == "MES":
        #     acq = botorch.acquisition.max_value_entropy_search.qMaxValueEntropy(
        #         model=model,
        #         candidate_set=torch.rand(self.params["candidate_set"]),
        #         num_fantasies=self.params["MES_num_fantasies"], 
        #         num_mv_samples=10, 
        #         num_y_samples=128,            
        #         )
        
        return acq

    def _init_MCsampler(self,num_samples):
        return botorch.sampling.samplers.SobolQMCNormalSampler(num_samples=num_samples)

    def _init_GPs(self,gp_name,gp_params):
        return GPs.BOtorch_GP(gp_name,**gp_params)