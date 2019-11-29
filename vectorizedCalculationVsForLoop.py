import numpy as np
import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("Vectorized calculation time is:" + str(1000 * (toc - tic)) + "ms.")

c = 0
tic =time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()
print("for_loop calculation time is:" + str(1000 * (toc - tic)) + "ms.")



import math

num = 10000
p = np.linspace(1,num,num)

tic = time.time()
q = np.zeros(p.size)
for i in range(p.size):
    q[i] = np.exp(p[i])
toc = time.time()
print("for_loop calculation time is:" + str(1000 * (toc - tic)) + "ms.")

tic = time.time()
q = np.zeros(p.size)
r = np.exp(p)
toc = time.time()
print("Vectorized calculation time is:" + str(1000 * (toc - tic)) + "ms.")
