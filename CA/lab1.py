import numpy as np
import matplotlib.pyplot as plt
import numpy.matrixlib as npm

n = 3
I = np.eye(n)
b = float(input("input b: "))
k = int(input("input k: "))
B = np.array([[0],
              [0],
              [b]])
C = np.array([[1., 0, 0]])

A = np.array([[0, 1., 0],
              [0, 0, 1.],
              [-1., -1., -3.]])

T = float(input("input T: "))
q = int(input("input q: "))
i = int(input("input iterations: " ))

x = np.array([[0],
              [0],
              [0]])

y = C.dot(x)

u1 = 1

u2 = -1

def Fq(q):
    F=I
    for i in range (q+1):
        F = F + (np.linalg.matrix_power(A.dot(T), i) )/(np.math.factorial(i))
    F = F - I
    return F

def Gq(q):
    G = (F - I)
    G= G.dot(np.linalg.inv(A))
    G= G.dot(B)
    return G

F = Fq(q)
G = Gq(q)

def first(y, x, F, G):
    global k
    i =0
    k = 0
    fig1 = plt.figure()
    plt.axis([0., i*T, -5, 5])
    plt.grid()
    fig1.set_size_inches(12, 8, forward=True)
    plt.xlabel("t")
    plt.ylabel("y")
    x1 = []
    x2 = []
    x3 = []
    yr = []
    for j in range(0, i):

        x = F.dot(x)  + (G)*u1
        y = C.dot(x)
        print(y)
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        k = k +1
    plt.title("First task")
    k = range(k)
    plt.plot(k, x1, label = 'x1')
    plt.plot(k, x2, label = 'x2')
    plt.plot(k, x3, label = 'x3')
    plt.plot(k, yr, label = 'y')
    plt.legend()
    plt.show()
    i = i*T

def second( y, x, F, G):
    global k
    k = 0
    fig2 = plt.figure()
    plt.axis([0., i, -5, 5])
    plt.grid()
    fig2.set_size_inches(12, 8, forward=True)
    plt.xlabel("t")
    plt.ylabel("y")
    x1 = []
    x2 = []
    x3 = []
    yr = []
    for j in range(0, int(i/2)):
        x = F.dot(x)  + (G)*u1
        y = C.dot(x)
        print(y)
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        k= k+1
    for j in range(int(i/2), int(i)):
        x = F.dot(x)  + (G)*u2
        y = C.dot(x)
        print(y)
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        k = k +1
    k = range(k)
    plt.plot(k, x1, label = 'x1')
    plt.plot(k, x2, label = 'x2')
    plt.plot(k, x3, label = 'x3')
    plt.plot(k, yr, label = 'y')
    plt.legend()
    plt.title("Second task")
    plt.show()

def third( y, x, F, G ):
    global k
    k = 0
    fig3 = plt.figure()
    plt.axis([0., i, -6, 6])
    plt.grid()
    fig3.set_size_inches(12, 9, forward=True)
    plt.xlabel("t")
    plt.ylabel("y")
    x1 = []
    x2 = []
    x3 = []
    yr = []
    for j in range(0, int(i/3)):
        x = F.dot(x)  + (G)*u1
        y = C.dot(x)
        print(y)
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        k= k+1
    for j in range(0, int(i/3)):
        x = F.dot(x)  + (G)*u2
        y = C.dot(x)
        print(y)
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        k= k+1
    for j in range(0, int(i/3)):
        x = F.dot(x)  + (G)*u1
        y = C.dot(x)
        print(y)
        print("k = %s, X1 = %s, X2 = %s, X3 = %s"%(k, x[0], x[1], x[2]))
        x1.append(float(x[0]))
        x2.append(float(x[1]))
        x3.append(float(x[2]))
        yr.append(float(y[0]))
        k= k+1
    k = range(k)
    plt.plot(k, x1, label = 'x1')
    plt.plot(k, x2, label = 'x2')
    plt.plot(k, x3, label = 'x3')
    plt.plot(k, yr, label = 'y')
    plt.legend()
    plt.title("Third task")
    plt.show()


first( y, x, F, G)
print("first finished")
second(y, x, F, G)
print("second finished")
third(y, x, F, G)
print("third finished")