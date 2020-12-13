import numpy as np
import matplotlib.pyplot as plt


def matrix_exp_func(accuracy):

    def matrix_exp(matrix):
        assert matrix.shape[0] == matrix.shape[1],\
            "You can calculate matrix exp only for square matrices."

        q = accuracy
        # E + A + (A^2)/2! + (A^3)/3! + ...
        res = np.eye(matrix.shape[0])
        for i in range(1, q):
            res += np.linalg.matrix_power(matrix, i) / np.math.factorial(i)
        return res

    return matrix_exp

# global parameters
def u(t: float):
    """ Control function.

    :param t - point of time.
    """
    return 1


a1 = 1
a2 = 3
b = 1

dt = 0.02

q = 10  # accuracy of numeric calculations
matrix_exp = matrix_exp_func(q)

l2 = 0
l3 = 0  # const

K = 1000    # number of iterations

# dx/dt = A*x(t) + B*u(t)
# y(t) = C*x(t)

# dim A = 3 x 3
A = np.array(
    [[0, 1, 0],
     [0, 0, 1],
     [-1, -a1, -a2]]
)

# dim B = 3 x 1
B = np.array(
    [[0],
     [0],
     [b]]
)

# dim C = 1 x 3
C = np.array(
    [[1, 0, 0]]
)



class LinearStaticSystem:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, p=None):
        self._A = A.copy()
        self._B = B.copy()
        self._C = C.copy()
        self._p = p if p is not None else np.zeros((1, self._A.shape[1]))
        self.F = self._fi_matrix()
        self.H = self._gamma_matrix()

        # inner state of system
        self._x = np.array(
            [[0],
             [0],
             [0]]
        )

    def _fi_matrix(self):
        return matrix_exp(self._A * dt)

    def _gamma_matrix(self):
        res = self.F - np.eye(self.F.shape[0])
        return res @ np.linalg.inv(self._A) @ self._B

    @property
    def y(self):
        return (self._C @ self._x)[0, 0]

    @property
    def x(self):
        return self._x

    def do_step(self, u_k: float):
        """
        Change current state of system from kth to (k+1)th.

        :param u_k: control value
        :return:
        """

        u_with_feedback = u_k - self._p @ self._x
        self._x = self.F @ self._x + self.H * u_with_feedback
        return self.y

    def clear(self):
        self._x = np.array(
            [[0],
             [0],
             [0]]
        )


# task: minimize function J(l2, l3) = |y0 - u0|*dt + |y1 - u1|*dt + ...


def J(param1, param2):
    p = np.array([[0, param1, param2]])
    system = LinearStaticSystem(A, B, C, p)
    ys = [system.y]
    us = []
    for k in range(K):
        t_k = dt*k
        us.append(u(t_k))
        ys.append(system.do_step(us[-1]))
    us.append(u(dt*K))
    res = 0
    for i in range(K+1):
        res += abs(ys[i] - us[i]) * dt
    return res


delta_l2 = 0.05
min_J = zero_J = J(l2, l3)

times = 0
for i in range(1000):  # find argmin J(l2) for l2 from [0; 50]
    times += 1
    l2 += delta_l2
    curr_J = J(l2, l3)
    if curr_J < min_J:
        min_J = curr_J
    # else:
    #     break

if times == 10000:
    print("timeout")

p_opt = np.array([[0, l2, l3]])


def buildGraphicks(system: LinearStaticSystem):
    system.clear()

    ys = [system.y]
    for k in range(K):
        t_k = dt * k
        u_k = u(t_k)
        ys.append(system.do_step(u_k))

    plt.xlabel('Time t')
    plt.ylabel('y(t)')
    time_list = [dt * i for i in range(K+1)]
    plt.plot(time_list, ys)
    plt.plot(time_list, [u(t) for t in time_list], 'r--')
    plt.gca().set_xlim([0, 20])
    plt.gca().set_ylim([0, 2])
    plt.show()


plt.title(f"Система без ответа. J = {zero_J}")
sys1 = LinearStaticSystem(A, B, C)
buildGraphicks(sys1)

plt.title(f"Система с ответом: L = (0, {l2: .2f}, {l3: .2f}). J = {min_J}")
sys2 = LinearStaticSystem(A, B, C, p_opt)
buildGraphicks(sys2)

