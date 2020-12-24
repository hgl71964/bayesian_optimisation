import threading
import numpy as np

# deadlock
l = threading.Lock()
print("before first acquire")
# l.acquire()
# print("before second acquire")
# l.acquire()
print("acquired lock twice")

print(np.random.rand(3))