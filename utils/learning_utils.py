
import numpy as np


def sigmoid(z):
    # return np.where(z >= 0, 1/(1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return 1 / (1 + np.exp(-z))


def softmax(z):
    # z = z - np.max(z, axis=-1)
    exps = np.exp(z)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def bce_loss(y, y_pred, epsilon=1e-10):
    # numerical stability
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return np.mean(-y * np.log2(y_pred) - (1-y) * np.log2(1 - y_pred), axis=0)


def cross_entropy_loss(y, y_pred, epsilon=1e-10):
    '''
    input is of size T x N x 10
    '''
    # numerical stability
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return np.mean(np.sum(-y * np.log2(y_pred), axis=-1), axis=-1)


def create_averages_array(arr):
    return np.cumsum(arr, axis=1) / np.arange(1, arr.shape[1] + 1)


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


# def predict(X, w, b):
#     # X --> Input.
#     # w --> weights.
#     # b --> bias.
#
#     # Predicting
#     z = X @ w + b
#     y_hat = softmax(z)
#
#     # Returning the class with highest probability.
#     return np.argmax(y_hat, axis=1)


def one_hot(idx, n_elements):
    assert np.max(idx) < n_elements
    if isinstance(idx, int):
        vec = np.zeros((n_elements, 1))
        vec[idx] = 1.
        return vec
    else:
        n_vecs = idx.shape[0]
        one_hots = np.zeros((n_vecs, n_elements))
        one_hots[np.arange(n_vecs), idx] = 1.
        return one_hots
