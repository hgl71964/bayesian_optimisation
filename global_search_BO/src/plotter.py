import matplotlib.pyplot as plt
import numpy as np
import torch as tr

class rosenbrock_plot:

    @staticmethod
    def plot(
            x: tr.tensor or np.ndarray,
            y: tr.tensor or np.ndarray,
            path: tr.tensor or np.ndarray, # [2, n]; row_2: x coor; row_2: y coor
            ):

        if isinstance(x, tr.Tensor):
            x, y, path = x.numpy(), y.numpy(), path.numpy()
            
        X, Y = np.meshgrid(x, y)
        f = lambda x,y: (1-x) ** 2 + 100*((y-x**2)**2)
        Z = f(X,Y)

        fig, ax = plt.subplots(figsize=(12,8))

        # ax.set_aspect('equal')
        cf = ax.contourf(X,Y,Z,np.arange(0, 10000, 1000),extend='both')
        ax.set_xlabel('x',fontsize = 16)
        ax.set_ylabel('y',fontsize = 16)
        # ax.set_title('Gradient descent',fontsize = 18)
        # fig.colorbar(cf, ax=ax)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.ax.tick_params(labelsize=14) 

        ax.scatter(path[0,:], path[1,:], marker='o', color='red', label='descent path')
        ax.scatter([1],[1],marker='o',color='white',label='global minimum')
        ax.legend(prop={'size':16})
        plt.show()

# if __name__ == "__main__":
#     x = tr.rand(2)

#     print(isinstance(x, tr.Tensor))