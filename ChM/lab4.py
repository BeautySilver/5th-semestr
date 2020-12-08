import math
import numpy as np

f = lambda x: 3*x**2 - math.cos(math.pi*x)**2 #целевое уравнение
x = np.array([x for x in np.arange(-1.8, 2.2, 0.4)])
#x[5] = 0
y = np.array([f(x) for x in x])
#print(x)
#print(y)

Lx = [0,0,0,0,0,0,0,0,0,0,0]
for i in range(1, 10):
    Lx[i-1] = ( f(x[i]) * x[i-1] - f(x[i-1]) * x[i] ) / ( f(x[i]) - f(x[i-1]) )

print(Lx)
res = np.array([f(lx) for lx in Lx])
res_x = np.argmin(np.abs(res))
print(Lx[res_x])