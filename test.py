from global_search_BO.es_interface import es_loop
import torch as tr
import numpy as np

print(es_loop)

a = np.ones(3)

print(tr.from_numpy(a).dtype)

print(tr.from_numpy(a.astype(np.float32)).dtype)