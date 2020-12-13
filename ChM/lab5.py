import numpy as np

func = lambda x: x ** 2 - 3 * x


def trap(a, b, n):
    res = 0
    x = np.linspace(a, b, n, endpoint=False)
    for i in range(0, n - 1):
        res = res + (x[i + 1] - x[i]) * (func(x[i]) + func(x[i + 1])) / 2
    return res


def Runge(a, b, n):
    print ("Estimate of Runge: ", (abs(trap(a, b, 2 * n) - trap(a, b, n)) / 3))
    return (abs(trap(a, b, 2 * n) - trap(a, b, n)) / 3)


print("integral = ", trap(1, 4, 3))

Runge(2, 6, 2)