# Adapted from https://github.com/Manaswip/MNIST-classification-without-libraries-using-neural-networks

import numpy as np
import scipy.io as sio
from scipy import optimize as o

import checkhint
from checkhint import _checkpoint

DATA_DIR = "MNISTData"

l1 = 1

hidden_layer_size = 300
num_labels = 10


def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def sigmoid_gradient(z):
    x = sigmoid(z)
    g = np.multiply(x, 1 - x)
    return g


def mk_grad(x, y, rows, cols):
    input_layer_size = rows * cols
    t1_size = hidden_layer_size * (input_layer_size + 1)
    t2_size = (hidden_layer_size + 1) * hidden_layer_size
    t3_size = (hidden_layer_size + 1) * num_labels

    def grad(params):
        nonlocal x
        nonlocal y

        theta1 = np.reshape(
            params[0:t1_size],
            ((input_layer_size + 1), hidden_layer_size),
        )

        theta2 = np.reshape(
            params[t1_size : t1_size + t2_size],
            ((hidden_layer_size + 1), hidden_layer_size),
        )

        theta3 = np.reshape(
            params[t1_size + t2_size :],
            ((hidden_layer_size + 1), num_labels),
        )

        theta1 = np.transpose(theta1)
        theta2 = np.transpose(theta2)
        theta3 = np.transpose(theta3)
        new_unrolled_theta = np.concatenate(
            [
                (np.transpose(theta1[:, 1:])).ravel(),
                (np.transpose(theta2[:, 1:])).ravel(),
                (np.transpose(theta3[:, 1:])).ravel(),
            ]
        )
        # Number of trainig data sets
        m = x.shape[0]
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)
        theta3_grad = np.zeros(theta3.shape)
        # Concatenating bias unit
        x_ = np.concatenate((np.ones((m, 1)), x), axis=1)

        # Number of features
        n = x_.shape[1]

        new_y = np.zeros((num_labels, m))

        for i in range(0, m):
            new_y[y.item(i), i] = 1
        #  	new_y[y.item(i),i] = 1;

        # Calculating delta terms using back propogation
        delta1_grad = np.zeros(theta1.shape)
        delta2_grad = np.zeros(theta2.shape)
        delta3_grad = np.zeros(theta3.shape)

        # back propogation algorithm
        for i in range(0, m):
            a1 = x[i, :][np.newaxis]
            a2 = sigmoid(np.dot(theta1, np.transpose(a1)))
            a2 = np.concatenate((np.ones((1, 1)), a2))
            a3 = sigmoid(np.dot(theta2, a2))
            a3 = np.concatenate((np.ones((1, 1)), a3))
            a4 = sigmoid(np.dot(theta3, a3))

            delta4 = a4 - np.transpose(new_y[:, i][np.newaxis])
            delta = np.dot((np.transpose(theta3)), delta4)
            delta3 = np.multiply(delta[1:], sigmoid_gradient(np.dot(theta2, a2)))
            delta = np.dot((np.transpose(theta2)), delta3)
            delta2 = np.multiply(
                delta[1:], sigmoid_gradient(np.dot(theta1, np.transpose(a1)))
            )

            delta1_grad = delta1_grad + np.dot(delta2, a1)
            delta2_grad = delta2_grad + np.dot(delta3, np.transpose(a2))
            delta3_grad = delta3_grad + np.dot(delta4, np.transpose(a3))

        theta1_grad = (1 / m) * delta1_grad
        theta2_grad = (1 / m) * delta2_grad
        theta3_grad = (1 / m) * delta3_grad
        # gradient of 3 layers
        theta1_grad[:, 1:] = theta1_grad[:, 1:] + (theta1[:, 1:]) * (l1 / m)
        theta2_grad[:, 1:] = theta2_grad[:, 1:] + (theta2[:, 1:]) * (l1 / m)
        theta3_grad[:, 1:] = theta3_grad[:, 1:] + (theta3[:, 1:]) * (l1 / m)
        # concatenate 3 layers gradient to one vector
        return np.concatenate(
            [
                (np.transpose(theta1_grad)).ravel(),
                (np.transpose(theta2_grad)).ravel(),
                (np.transpose(theta3_grad)).ravel(),
            ]
        )

    return grad


def mk_cost(x, y, rows, cols):
    input_layer_size = rows * cols
    t1_size = (input_layer_size + 1) * hidden_layer_size
    t2_size = (hidden_layer_size + 1) * hidden_layer_size
    t3_size = (hidden_layer_size + 1) * num_labels

    def cost(params):
        nonlocal x, y

        theta1 = np.reshape(
            params[0:t1_size],
            ((input_layer_size + 1), hidden_layer_size),
        )

        theta2 = np.reshape(
            params[t1_size: t1_size + t2_size],
            ((hidden_layer_size + 1), hidden_layer_size),
        )

        theta3 = np.reshape(
            params[t1_size + t2_size :],
            ((hidden_layer_size + 1), num_labels),
        )

        theta1 = np.transpose(theta1)
        theta2 = np.transpose(theta2)
        theta3 = np.transpose(theta3)
        new_unrolled_theta = np.concatenate(
            [
                (np.transpose(theta1[:, 1:])).ravel(),
                (np.transpose(theta2[:, 1:])).ravel(),
                (np.transpose(theta3[:, 1:])).ravel(),
            ]
        )
        m = x.shape[0]
        x_ = np.concatenate((np.ones((m, 1)), x), axis=1)

        # Number of features
        n = x_.shape[1]
        # transforming y to new matrix with 10xm size
        new_y = np.zeros((num_labels, m))

        for i in range(0, m):
            new_y[y.item(i), i] = 1

        first_layer_activation = x_
        # Second layer activation function
        second_layer_activation = sigmoid(np.dot(theta1, np.transpose(x)))
        # appending bias unit
        second_layer_activation = np.concatenate(
            (np.ones((1, m)), second_layer_activation)
        )
        # Activation in third layer
        third_layer_activation = sigmoid(np.dot(theta2, second_layer_activation))
        # appending bias unit
        third_layer_activation = np.concatenate(
            (np.ones((1, m)), third_layer_activation)
        )
        hyp_function = sigmoid(np.dot(theta3, third_layer_activation))

        first_half = np.sum(np.multiply(new_y, np.log(hyp_function)))
        second_half = np.sum(np.multiply((1 - new_y), np.log(1 - hyp_function)))

        return ((-1.0 / m) * (first_half + second_half)) + (
            l1 / (2 * m) * (np.sum(np.multiply(new_unrolled_theta, new_unrolled_theta)))
        )

    return cost


@checkhint.schedule_checkpoints()
def train(rows, cols, cost, grad):
    input_layer_size = rows * cols
    t1_size = hidden_layer_size * (input_layer_size + 1)
    t2_size = (hidden_layer_size + 1) * hidden_layer_size
    t3_size = (hidden_layer_size + 1) * num_labels
    params = np.zeros((t1_size + t2_size + t3_size,))

    for j in range(0, 5):
        _iterations: 5
        for _ in range(0, 50):
            _iterations: 50
            params = o.fmin_cg(cost, params, fprime=grad, maxiter=1)
            _checkpoint()
        _checkpoint()

    t1 = np.reshape(
        params[0:t1_size],
        ((input_layer_size + 1), hidden_layer_size),
    )
    t2 = np.reshape(
        params[t1_size : t1_size + t2_size],
        ((hidden_layer_size + 1), hidden_layer_size),
    )
    t3 = np.reshape(
        params[t1_size + t2_size :],
        ((hidden_layer_size + 1), num_labels),
    )

    t1 = np.transpose(t1)
    t2 = np.transpose(t2)
    t3 = np.transpose(t3)

    return t1, t2, t3


if __name__ == "__main__":
    import os
    import array
    import struct

    def extract_data(prefix):
        img_data = os.path.join(".", f"{prefix}-images-idx3-ubyte")
        lbl_data = os.path.join(".", f"{prefix}-labels-idx1-ubyte")

        file_img = open(img_data, "rb")
        magic_nr, size, rows, cols = struct.unpack(">IIII", file_img.read(16))
        img = array("b", file_img.read())
        file_img.close()

        file_lbl = open(lbl_data, "rb")
        magic_nr, size = struct.unpack(">II", file_lbl.read(8))
        lbl = array("B", file_lbl.read())
        file_lbl.close()

        digits = np.arange(10)

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)

        images = np.zeros((N, rows * cols), dtype=np.uint8)
        labels = np.zeros((N, 1), dtype=np.uint8)

        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols : (ind[i] + 1) * rows * cols])
            labels[i] = lbl[ind[i]]

        return images, labels, rows, cols

    os.chdir(DATA_DIR)
    x, y, rows, cols = extract_data("train")
    x_test, y_test, rows_test, cols_test = extract_data("t10k")

    cost = mk_cost(x, y, rows, cols)
    grad = mk_grad(x, y, rows, cols)

    t1, t2, t3 = train(rows, cols, cost, grad)

    m_test = x.shape[0]
    x_test = np.concatenate((np.ones((m_test, 1)), x_test), axis=1)

    first_layer_activation = x_test

    second_layer_activation = sigmoid(np.dot(t1, np.transpose(x_test)))
    second_layer_activation = np.concatenate(
        (np.ones((1, m_test)), second_layer_activation)
    )

    third_layer_activation = sigmoid(np.dot(t2, second_layer_activation))
    third_layer_activation = np.concatenate(
        (np.ones((1, m_test)), third_layer_activation)
    )

    hypothesis = sigmoid(np.dot(t3, third_layer_activation))

    predict = np.argmax(hypothesis, 0)
    predict = predict[np.newaxis]
    predict = np.transpose(predict)

    a = predict == y_test
    np.uint8(a)

    print("Training Accuracy:", np.mean(a) * 100)
