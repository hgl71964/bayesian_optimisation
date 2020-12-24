import os
import numpy as np
from global_search_BO.interface_global import bayes_opt_interface 

print(os.getcwd())

a = np.random.rand(2,)

print(isinstance(a, np.ndarray))
print(isinstance(a, tuple))

