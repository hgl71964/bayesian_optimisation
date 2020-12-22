import matplotlib.pyplot as plt
import numpy as np
import torch as tr


class saver:

    @staticmethod
    def standard_save(x, y, r0, name, path):

        data = {}
        data["name"] = name
        data["x"] = x
        data["y"] = y
        data["r0"] = r0
        
        tr.save(data, path)
        print(f"saving {name} data")