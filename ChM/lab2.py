import numpy as np
 #Метод Гуса и Якоби

A = np.array([[10, 9, 3, 4],
              [9,11, 5, 6],
              [3, 5, 8, 6],
              [4, 6, 6, 7]])


b = np.array([7.7, 8.96, 99, 1])

eps = 0.00001

x0 = np.array([0., 0., 0., 0.])
end = False
iterations = 1


while not end:
    x1 = np.copy(x0)
    x1[0] = (-A[0, 1] * x0[1] - A[0, 2] * x0[2] - A[0, 3] * x0[3] + b[0]) / A[0, 0]
    x1[1] = (-A[1, 0] * x1[0] - A[1, 2] * x0[2] - A[1, 3] * x0[3] + b[1]) / A[1, 1]
    x1[2] = (-A[2, 0] * x1[0] - A[2, 1] * x1[1] - A[2, 3] * x0[3] + b[2]) / A[2, 2]
    x1[3] = (-A[3, 0] * x1[0] - A[3, 1] * x1[1] - A[3, 2] * x1[2] + b[3]) / A[3, 3]
    end = max(abs(x1-x0)) < eps
    x0 = x1
    iterations += 1


print('final')
print(x1)
print(np.linalg.solve(A,b))

det = np.linalg.det(A)
norm1 = np.linalg.norm(A, ord=np.inf)
norm2 = np.linalg.norm(np.linalg.inv(A), ord=np.inf)
M = norm1 * norm2
print(f'Определитель: {det}\nЧисло обумовленості: {M}')