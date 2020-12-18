"""
2020 Summer internship

Implement experiment results && other utils
"""
import os
import torch
import copy
from Botorch_opt import bayesian_optimiser
from function_slicer import slicer
import time

class exp_helper:


    @staticmethod
    def exp0(gp_name: str,
            gp_params: dict,
            hyper_params: dict,
            data_folder: str,
            n: int,  # portfolio number
            T: int,
            x0: torch.Tensor,
            y0: torch.Tensor,
            m0: float,
            api: callable,
            exp_repitition: int):
        """
        random search
        """
        res = {}
        total_dim = x0.size(-1)

        #  to decorate multi-run
        bayes_loop = exp_helper.multi_run_decorator(slicer.static_random_query, runs=exp_repitition)

        #  experiment might fail because of numerical issue from cholesky decomposition
        for _ in range(2):
            try:
                x, y = bayes_loop(total_dim, api, m0, N=150, q=10)
            except Exception as err:
                print("")
                print(f"{err} occurs during experiment")
                print("")
                time.sleep(60)
            else:
                break

        algo = "random"
        res[algo] = (x, y, m0)
        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} random")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res
        return None

    @staticmethod
    def exp1(gp_name: str,
            gp_params: dict,
            hyper_params: dict,
            data_folder: str,
            n: int,  # portfolio number
            T: int,
            x0: torch.Tensor,
            y0: torch.Tensor,
            m0: float,
            api: callable,
            exp_repitition: int):
        """
        the first experiment; exploration-exploitation trade-off
        """
        print("starting first experiment")

        betas = [0.1, 0.5, 1, 2, 10]
        res = {}
        temp_max = -float("inf")
        
        for beta in betas:
            hyper_params["beta"], hyper_params["acq_name"] = beta, "UCB"
            bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)

            #  to decorate multi-run
            bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop_hard_termination, runs=exp_repitition)

            #  experiment might fail because of numerical issue from cholesky decomposition
            for _ in range(2):
                try:
                    x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size=1)
                except Exception as err:
                    print("")
                    print(f"{err} occurs during experiment")
                    print("")
                    time.sleep(60)
                else:
                    break
            
            temp = sum(y).max()
            if temp > temp_max:
                temp_max = temp
                best_beta = beta

            algo = "UCB" + "_" + str(beta)
            res[algo] = (x, y, m0) 
            
        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} UCBs")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res

        return best_beta
    
    @staticmethod
    def exp2(gp_name: str,
            gp_params: dict,
            hyper_params: dict,
            data_folder: str,
            n: int,  # portfolio number
            T: int,
            x0: torch.Tensor,
            y0: torch.Tensor,
            m0: float,
            api: callable,
            exp_repitition: int,
            beta:float, 
            ):

        """
        exp2 to test common acquisition functions & their parallelism
        """
        print("starting exp2")

        acq_funcs = ["qUCB", "EI", "qEI"]
        res = {} 

        for acq_func in acq_funcs:
            hyper_params["acq_name"] =  acq_func
            batch_size = 2 if acq_func[0] == "q" else 1  # choose parallelism accordingly
            bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)

            #  to decorate multi-run
            bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop_hard_termination, runs=exp_repitition)

            #  experiment might fail because of numerical issue from cholesky decomposition
            for _ in range(2):
                try:
                    x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size)
                except Exception as err:
                    print("")
                    print(f"{err} occurs during experiment")
                    print("")
                    time.sleep(60)
                else:
                    break

            if acq_func[-1] == "B":
                algo = acq_func + "_" + str(beta)
            else:
                algo = acq_func
            res[algo] = (x, y, m0) 

        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} UCB_EI")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res
        return None
        

    @staticmethod
    def exp3(gp_name: str,
            gp_params: dict,
            hyper_params: dict,
            data_folder: str,
            n: int,  # portfolio number
            T: int,
            x0: torch.Tensor,
            y0: torch.Tensor,
            m0: float,
            api: callable,
            exp_repitition: int):

        """
        exp3 to test KG
        """
        print("starting exp3")
        res = {} 
        hyper_params["acq_name"] =  "qKG"
        batch_size = 1
        bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)

        #  to decorate multi-run
        bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop_hard_termination, runs=exp_repitition)

        #  experiment might fail because of numerical issue from cholesky decomposition
        for _ in range(2):
            try:
                x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size)
            except Exception as err:
                print("")
                print(f"{err} occurs during experiment")
                print("")
                time.sleep(60)
            else:
                break

        res["qKG"] = (x, y, m0)  

        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} qKG")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res
        return None
    
    @staticmethod
    def exp4(gp_name: str,
            gp_params: dict,
            hyper_params: dict,
            data_folder: str,
            n:int,  # portfolio number
            T: int,
            x0: torch.Tensor,
            y0: torch.Tensor,
            m0: float,
            api: callable,
            exp_repitition: int):

        """
        exp4 to use warmup phase
        """
        print("starting exp4")
        res = {} 

        #  rebalancing exploration-exploitation trade-off
        hyper_params["acq_name"], hyper_params["beta"] =  "UCB", 10
        bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)
        bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop_warmup1, runs=exp_repitition)

        #  experiment might fail because of numerical issue from cholesky decomposition
        for _ in range(2):
            try:
                x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size=1)
            except Exception as err:
                print("")
                print(f"{err} occurs during experiment")
                print("")
                time.sleep(60)
            else:
                break
        res["warmup1"] = (x, y, m0) 

        #  switching to KG
        hyper_params["acq_name"], hyper_params["beta"] =  "UCB", 10
        bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)
        bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop_warmup2, runs=exp_repitition)

        #  experiment might fail because of numerical issue from cholesky decomposition
        for _ in range(2):
            try:
                x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size=1)
            except Exception as err:
                print("")
                print(f"{err} occurs during experiment")
                print("")
                time.sleep(60)
            else:
                break
        res["warmup2"] = (x, y, m0) 
        
        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} warmups")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res
        return None


    @staticmethod
    def tuning_method_exp(gp_name: str,
                        gp_params: dict,
                        hyper_params: dict,
                        data_folder: str,
                        n:int,  # portfolio number
                        T: int,
                        x0: torch.Tensor,
                        y0: torch.Tensor,
                        m0: float,
                        api: callable,
                        exp_repitition: int
                        ):
        hyper_params["beta"] = 1
        hyper_params["acq_name"] =  "UCB"
        res = {} 
        opts = ["ADAM", "quasi_newton"]

        for opt in opts:
            gp_params["opt"] = opt
            bayes_opt = bayesian_optimiser(gp_name, gp_params, hyper_params)

            #  to decorate multi-run
            bayes_loop = exp_helper.multi_run_decorator(bayes_opt.outer_loop, runs=exp_repitition)

            #  experiment might fail because of numerical issue from cholesky decomposition
            for _ in range(5):
                try:
                    x, y = bayes_loop(T, copy.deepcopy(x0), copy.deepcopy(y0), m0, api, batch_size=1)
                except Exception as err:
                    print("")
                    print(f"{err} occurs during experiment")
                    print("")
                    time.sleep(60)
                else:
                    break

            algo = f"{opt} UCB_1"
            res[algo] = (x, y, m0) 

        #  save result 
        full_path = os.path.join(data_folder, f"portfolio_{n} tuning_method")
        with open(full_path + ".pt", "wb") as f:
            torch.save(res, f)
            f.close()
        del res
        return None


    @staticmethod
    def slices2train( slices, ndim):
        """
        convert slices -> training points;

        Args:
            slices: snction_slicer
            ndim: input dimension; int

        Returns:
            X: tensor, shape [n,d]
            Y: tensor, shape [n,1]
        """

        for i, s in enumerate(slices):
            x, y, rewards, dim_vals, _, interest_dim = s

            query = torch.full( (x.size(0) * x.size(-1), ndim), dim_vals, dtype = torch.float)

            x = x.reshape(-1,1)      
            y = y.reshape(-1,1)
            rewards = rewards.reshape(-1,1)
            query[:, interest_dim] = torch.cat([x, y], dim=1)

            if i == 0:
                X = query
                Y = rewards
            else:
                X = torch.cat([X, query], dim = 0)
                Y = torch.cat([Y, rewards], dim = 0)
        return X, Y    

    @staticmethod
    def multi_run_decorator(func: callable,
                            runs: int):
        
        def wrapper(*args, **kwargs):
            xs = [None] * runs
            ys = [None] * runs

            for run in range(runs):    
                x, y = func(*args, **kwargs)
                xs[run] = x
                ys[run] = y

            return xs, ys
        return wrapper
        
