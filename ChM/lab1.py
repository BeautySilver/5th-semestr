#!/usr/bin/env python
# coding=utf8

#  Лабораторна робота No.1, Солоного Олега, САТР-3, дихотомія + мод. Ньютона
import numpy as np
import math

E = float(format(0.001, '.4f'))

func = lambda x: (x ** 5 - 5 * x + 2)

func_pohidna = lambda x: 5 * (x ** 4) - 5

func_dr_pohidna = lambda x: 20 * (x ** 3)


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



#iter(x0, phi)
#print(res1)
#print(apr_n_dihot)
#print(res2)
def metod_Newton():
    a = float(format(0.0, '.6f'))
    b = float(format(1.0, '.6f'))
    x = (a + b) / 2
    x = float(format(x, '.6f'))

    if func(x) * func(a) < 0:
        x0 = a
    else:
        x0 = b
    print
    "Умова: %s" % (func(x0) * func_dr_pohidna(x0))
    m1 = min(abs(float(format(func_pohidna(x0), '.6f'))), abs(float(format(func_pohidna(x), '.6f'))))
    M2 = max(abs(float(format(func_dr_pohidna(x0), '.6f'))), abs(float(format(func_dr_pohidna(x), '.6f'))))
    q = M2 * abs(x0 - x) / (2 * m1)
    q = float(format(q, '.6f'))
    apr_est = math.ceil(math.log(((math.log(abs(x - x0) / E) / math.log(1 / q)) + 1), 2))

    print("Початкові умови: x0 = %s, x = %s, m1 = %s, M2= %s, q = %s, апріорна оцінка = %s\n" % (
    x0, x, m1, M2, q, apr_est))

    for i in range(1, int(apr_est)):
        x = x - func(x) / func_pohidna(x)
        print("На кроці %s х = %s\n" % (i, x))


print
"\n\n dihotomiya\n\n"

metod_dihotomii()

print
"\n\n Newton\n"

metod_Newton()
