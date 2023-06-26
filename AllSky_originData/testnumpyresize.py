import numpy as np
import time

start = time.time()
a = np.zeros(int(1e8))
print(time.time()-start)
for i in range(int(1e8)):
    a.resize(len(a)+1)
print(time.time()-start)
