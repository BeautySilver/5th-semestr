import matplotlib.pyplot as plt
import numpy as np
import math


class program:
    __iterations = 0

    def __init__(self, iterations=0):
        self.__iterations = iterations

    def P(self, F):
        P_next = np.identity(3)
        for index in range(self.__iterations):
            P_previous = P_next
            P_next = np.dot(F, P_previous)
        return P_next

    def S(self, F, F_Inverse, G, collection):
        P_next = np.identity(3)
        for index in range(self.__iterations):
            P_previous = P_next
            P_next = np.dot(F_Inverse, P_previous)
            collection.append(np.dot(P_next, G))
        sum_next = np.zeros((3, 3))
        for index in range(1, self.__iterations):
            sum_previous = sum_next
            sum_next = sum_previous + \
                np.dot(collection[index], collection[index].transpose())
        return sum_next

    def calculate(self, a_1, a_2, b, q, T_0, x_1, x_2, x_3, special_coordinate):
        A = np.matrix([
            [0, 1, 0],
            [0, 0, 1],
            [-1, -a_1, -a_2]
        ], dtype=float)
        B = np.matrix([
            [0],
            [0],
            [b]
        ])
        C = np.matrix([
            [1, 0, 0]
        ], dtype=float)
        F = np.identity(3)
        F_Inverse = np.identity(3)
        for index in range(1, q + 1):
            F += np.linalg.matrix_power(A * T_0, index) / \
                float(math.factorial(index))
        for index in range(1, q + 1):
            if index % 2 != 0:
                F_Inverse -= np.linalg.matrix_power(
                    A * T_0, index) / float(math.factorial(index))
            else:
                F_Inverse += np.linalg.matrix_power(
                    A * T_0, index) / float(math.factorial(index))
        vector = np.matrix([
            [x_1],
            [x_2],
            [x_3]
        ], dtype=float)
        collection = []
        x_special = np.matrix([
            [special_coordinate],
            [0],
            [0]
        ], dtype=float)
        # claculate Г
        G = np.dot(np.dot((F - np.identity(3)), np.linalg.inv(A)), B)

        L = np.dot(self.P(F), self.S(F, F_Inverse, G, collection))
        L_Inverse = np.linalg.inv(L)
        l_0 = np.dot(L_Inverse, x_special)
        x_next = np.dot(F, vector) + (G * 0)
        array_of_vectors = []
        array_of_vectors.append(x_next)
        u_next = np.dot(collection[0].transpose(), l_0)
        array_X = []
        array_Y = []
        array_Z = []
        for index in range(self.__iterations):
            u_previous = u_next
            x_previous = x_next
            x_next = np.dot(F, x_previous) + (G * u_previous)
            u_next = np.dot(collection[index].transpose(), l_0)
            array_Y.append(float(np.dot(C, x_next)))
            array_of_vectors.append(x_next)
            array_X.append(index)
            array_Z.append(u_next)
        return array_X, array_Y, array_of_vectors, array_Z


def draw_parameters(array_X, arrays_Y, management, names):
    plots = [[], [], [], []]
    for index in range(len(array_X)):
        plots[0].append(float(arrays_Y[index][0]))
        plots[1].append(float(arrays_Y[index][1]))
        plots[2].append(float(arrays_Y[index][2]))
        plots[3].append(float(management[index]))
    plt.plot(array_X, plots[0], label=names[0])
    plt.plot(array_X, plots[1], label=names[1])
    plt.plot(array_X, plots[2], label=names[2])
    plt.plot(array_X, plots[3], label=names[3])
    plt.legend()
    plt.show()
    return


# a_1, a_2, b, q, T_0, x_1, x_2, x_3, special_coordinate
if __name__ == "__main__":
    b =  float(input("input b: "))
    a1 = float(input("input a1: "))
    a2 = float(input("input a2: "))
    T = float(input("input T: "))
    q = int(input("input q: "))
    x_0_aim = float ( input ("input x_0_aim: "))
    a, b, c, z = program(110).calculate(
        a1, a2, b, q, T, 0, 0, 0, x_0_aim)
    draw_parameters(a, c, z, ['x1', 'x2', 'x3', 'u'])

