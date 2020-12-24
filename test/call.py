import os
import numpy as np
import concurrent.futures 
from time import sleep
import torch as tr


def some_func(name, number):
    print(name)
    sleep(1)
    print(number)
    sleep(1)
    return name

if __name__ == "__main__":
    a=tr.empty((10, 1))
    for i in range(10):
        a[i]=i 

    print(a)
