
"""
2020 Summer internship

a bayes optimiser agent to encapsulate model, acquisition function and environment
"""
import aquisition_func.GP_EI as GP_EI
import GPs.One_dim_GPR as One_dim_GPR
import aquisition_func.KG
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import copy


class bayes_optimiser:
    def __init__(self,**kwargs):
        super().__init__()
        """
        a bayesian optimiser

        Args:
            kwargs: consists of hyper-parameter for model, acquisition function & environment respectively

        Attrs:
            model: a Gaussian process regression for function approximation

            acquisition function: used to sample exficiently 

            env: propose query, get rewards
        """
        self.acquisition_function = GP_EI._expected_improvement(**kwargs)
        self.model = One_dim_GPR._GP_regression(**kwargs)
        self.env = environment(**kwargs)

    def decision_process(self,bounds,opt_restart,x,y,X_test):
        """
        a sequential decision-making process; provide a query and get a reward;

        """

        x_sample = x
        y_sample = y

        for _ in range(10):
            query = self.acquisition_function.propose_position(self.model,bounds,opt_restart,x,y,X_test,mean_function=0,mean_function_test=0)
            reward = self.env.transition(query)

            x_sample = np.append(x_sample,query)
            y_sample = np.append(y_sample,reward)

        print(y_sample)


    def plot_decision(self,bounds,opt_restart,x,y,X_test,mean_function=0,mean_function_test=0):
        """
        plot model and acquisition function as well as next query at thie time step

        Warnings:
            we only plot 1d function, otherwise raises error
            
        Args:
            x: x_sample we have thus far
            y: corrsponding y sampples
            X_test: testing points to smooth out the posterior

            bounds, opt_restart see acquisition function
        """
        if len(X_test.shape)!=1:
            raise ValueError("can only visualise 1d")
        
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        ax1,ax2 = ax.flatten()

        _,query = self.acquisition_function.propose_position(self.model,bounds,opt_restart,x,y,X_test)

        pst_mean, pst_Sigma = self.model.gp_posterior(x,y,X_test,mean_function,mean_function_test)
        pst_std = np.sqrt(np.diag(pst_Sigma)).flatten()

        # function samples; N = 5
        sample_y = np.random.multivariate_normal(mean=pst_mean, cov=pst_Sigma, size=3)

        ###### GP ########
        ax1.set_ylabel('y',fontsize = 12)
        ax1.set_xlabel('x',fontsize = 12)
        ax1.set_title('Gaussian process posterior',fontsize = 14)
        ax1.set_xlim(0,1)
        ax1.margins(x=0.5)
        

        ax1.scatter(x,y,marker='o',
            linewidth=4,color='black', label='samples', alpha=0.9)

        ax1.plot(X_test,pst_mean,linewidth=2,color='blue', label='posterior mean', alpha=1)

        ax1.plot(X_test, sample_y.T, '-', alpha=0.3)

        ax1.fill_between(X_test, pst_mean-2*pst_std, pst_mean+2*pst_std, color='red', 
                        alpha=0.2, label='$2 \sigma_{2|1}$')
        ax1.axvline(query,linewidth = 1,alpha=0.6,label="query")

        try: # plot the ground true curve if it is available
            ax1.plot(X_test,self.env.transition(X_test),linewidth=2,color="green",linestyle="--",
                label="ground true",alpha=1)
        except:
            pass
        ax1.legend()

        ###### Acquisition ########

        ax2.set_ylabel('y',fontsize = 12)
        ax2.set_xlabel('x',fontsize = 12)
        ax2.set_title('Acquisition Function',fontsize = 14)
        ax2.set_xlim(0,1)
        ax2.margins(x=0.5)

        ax2.plot(X_test, self.acquisition_function.compute_func(self.model,x,y,X_test,mean_function,mean_function_test),
            marker='o', color='red', linewidth=2,label= 'Expected improvement')
        ax2.axvline(query,linewidth = 1,alpha=0.6,label="query")
        ax2.legend()
        plt.show()


class environment:
    def __init__(self,**kwargs):
        self.x = np.linspace(0,1,50) # use to smooth out the function
        self.x_init = np.array([0.3,0.6]) # initial samples

    def function(self,x):
        return np.sin(10*x) + np.exp(x)/2

    def transition(self,x):
        return self.function(x)


if __name__ == '__main__':
    # initial samples
    env = environment()
    x_sample = copy.deepcopy(env.x_init)
    y_sample = env.function(x_sample)
    X_test = np.linspace(0,1,100)

    # hyper-parameter
    hyper_params = {"kernel_type":"SE",
                    "len_scale":5e-2,
                    "amplify_coef":8e-1,
                    "xi":1e-2,
                    }

    bounds=np.array([[0,1]])
    opt_restart = 10

    bayes_opt = bayes_optimiser(**hyper_params)

    bayes_opt.plot_decision(bounds,opt_restart,x_sample,y_sample,X_test)
    # print(bayes_opt.acquisition_function.propose_position(bayes_opt.model,bounds,opt_restart,x_sample,y_sample,X_test))

    # print('done')

    


