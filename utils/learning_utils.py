
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
    if len(arr.shape) == 2:
        return np.cumsum(arr, axis=0) / np.arange(1, arr.shape[0] + 1).reshape(-1, 1)
    if len(arr.shape) == 3:
        return np.cumsum(arr, axis=0) / np.arange(1, arr.shape[0] + 1).reshape(-1, 1, 1)
    else:
        raise NotImplementedError


def compute_OLS_estimator(trajectory):
    '''

    Args:
        trajectory: array of observations of size dxt

    Returns: compute the OLS estimator based on the given (full) trajectory

    '''
    # Compute OLS
    cov = trajectory[:, :-1] @ trajectory[:, :-1].T
    cross_cov = trajectory[:, 1:] @ trajectory[:, :-1].T
    A_ols = np.linalg.inv(cov) @ cross_cov
    return A_ols


def compute_OLS_error_arr(trajectory, A_star, start_from=2, subsample=1):
    '''

    Args:
        trajectory: array of observations of size dxT

    Returns: compute the OLS estimator array for every partial trajectory

    '''
    error_arr = []
    T = trajectory.shape[1]
    for t in range(start_from, T+1, subsample):
        if t % 100000 == 0:
            print(t, " OLS iterations passed !")
        A_ols_t = compute_OLS_estimator(trajectory[:, :t+1])
        error_t = np.linalg.norm(A_ols_t - A_star, ord=2)
        error_arr.append(error_t)
    return np.stack(error_arr)


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


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
