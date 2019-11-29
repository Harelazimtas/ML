import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(x, y, w1, w2, feature):
    normal = np.dot(feature, w1)
    grad = np.dot(feature, w2)
    plt.plot(x, normal, "d", label="normal equation")
    plt.plot(x, grad, "y", label="gradient")
    plt.plot(x, y, "r", label='linear')
    # plt.scatter(x, y)
    plt.legend()
    plt.show()


def create_feature(x, max_poly_deg):
    feature = np.zeros_like(x)
    for power in range(0, max_poly_deg+1):
        if power == 0:
            feature = x ** power
        else:
            x2 = x ** power
            feature = np.column_stack((feature, x2))
    return feature


def normal_equation(x, y, matrix):
    x_trans = np.transpose(matrix)
    A = np.linalg.inv(np.dot(x_trans, matrix))
    B = np.dot(x_trans, y)
    w = np.dot(A, B)
    return w


def gradient(x, y, epsilon_number, lambda1, matrix):
    # random
    list_of_w = np.zeros_like(matrix[0])
    norma = lambda1 + 1
    m = int(matrix.size/matrix[0].size)
    while norma > lambda1:
        gradient1 = np.zeros_like(matrix[0])
        for index in range(0, m):
            yp = np.dot(matrix[index],  list_of_w)
            tp = y[index]
            gradient1 += -(tp-yp) * 2 * matrix[index]
        gradient1 = gradient1 * 1/m
        list_of_w += gradient1 * (-epsilon_number)
        norma = np.linalg.norm(gradient1)
    return list_of_w


def main():
    max_poly_deg = 2
    x = np.load('x_train_2.npy')
    y = np.load('y_train_2.npy')
    y = np.reshape(y, (len(y), 1))
    matrix = np.reshape(x, (len(x), 1))
    feature = create_feature(matrix, max_poly_deg)
    # normal equation
    w1 = normal_equation(x, y, feature)
    # GD regression
    lambda1 = 0.0001
    epsilon_number = 0.1
    w2 = gradient(x, y, epsilon_number, lambda1, feature)
    plot_prediction(x, y, w1, w2, feature)


main()
"""
end
"""