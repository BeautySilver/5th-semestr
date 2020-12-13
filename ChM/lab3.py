import  math
import numpy as np

equation = lambda x: 3*x**2 - math.cos(math.pi*x)**2 #целевое уравнение
eps = 0.000001 #заданная точность
a = 1
b = 2
x0 = 1.5
phi = lambda x: 6*x + math.sin(math.pi*x)*math.cos(math.pi*x)
phi1 = lambda x: 3*(math.sin(math.pi*x))**2 - 3*math.cos(math.pi*x)**2 + 6

def itera(x0,phi,eps):
    res0 = phi(x0)
    res1 = phi(res0)
    n = 2
    while abs(res1-res0) >= eps:
        res0 = phi(res1)
        res1 = phi(res0)
        n += 1

    return res0, res1, n


def iter(x0, phi):
    res = phi(x0)
    for i in range(16):
        res = phi(res)
        print(res)

print ("Метод простых итераций")
iter(x0, phi)