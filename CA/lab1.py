import numpy as np
import numpy.linalg as ag
import matplotlib.pyplot as plt
import math

# Program settings
delta = 0.01
maxT = 20
steps = maxT / delta

# Functions
def matrix_exponencial(A:np.matrix, t):
    return np.eye(3) + A*t + ag.matrix_power(A*t, 2) / math.factorial(2) #+ A*t + ag.matrix_power(A*t, 3) / math.factorial(3) + + A*t + ag.matrix_power(A*t, 4) / math.factorial(4) + A*t + ag.matrix_power(A*t, 5) / math.factorial(5)


def u(step, utype):
    if utype == 1:
        return 1
    if utype == 2:
        if step < steps / 2:
            return 1
        else:
            return -1
    if utype == 3:
        if step % 2 == 1:
            return 1
        else:
            return -1
    if utype == 4:
        return 5


def Fi(A:np.matrix, delta):
    return matrix_exponencial(A, delta)


def H(Fi:np.matrix, A:np.matrix, B:np.matrix):
    return (Fi - np.eye(3)).dot(ag.matrix_power(A, -1)).dot(B)


# Parameters
a0 = 1
a1 = 2
a2 = 4
utype = 2
A = np.matrix([[0, 1, 0],
               [0, 0, 1],
               [-a0, -a1, -a2]])
B = np.matrix([[0],
               [0],
               [1]])
C = np.matrix([1, 10, 80])

x1 = 2
x2 = 0
x3 = 0
x1Array = [x1]
x2Array = [x2]
x3Array = [x3]
yArray = [C.dot(np.matrix([[x1Array[0]],
                           [x2Array[0]],
                           [x3Array[0]]]))]
tArray = [0]
Fi = Fi(A, delta)
H = H(Fi, A, B)

# Calculating
j = 0
i = 1
while j < maxT:
    # Calculating xi
    xi = np.matrix([[x1Array[i-1]],
                   [x2Array[i-1]],
                   [x3Array[i-1]]])

    # Creating y(t) values
    yArray.append(C.dot(xi))

    # Creating series (labeled x1(t)/x2(t)/x3(t))
    x1Array.append((Fi.dot(xi) + H * u(i, utype)).item(0))
    x2Array.append((Fi.dot(xi) + H * u(i, utype)).item(1))
    x3Array.append((Fi.dot(xi) + H * u(i, utype)).item(2))
    i = i + 1

    # Creating abscissa axis values (labeled t)
    tArray.append(j + delta)
    j = j + delta


# Drawing plots
plt.xlabel('Time t')
plt.ylabel('y(t)')
scatter1 = plt.scatter(tArray, yArray, s=1)
plt.show()

plt.xlabel('Time t')
plt.ylabel('x1(t)')
scatter2 = plt.scatter(tArray, x1Array, s=1)
plt.show()

plt.xlabel('Time t')
plt.ylabel('x2(t)')
scatter3 = plt.scatter(tArray, x2Array, s=1)
plt.show()

plt.xlabel('Time t')
plt.ylabel('x3(t)')
scatter4 = plt.scatter(tArray, x3Array, s=1)
plt.show()