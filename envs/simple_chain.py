
import numpy as np
from utils.prob_utils import sample_discrete, draw_in_l2_ball
# from keras.datasets import mnist
from utils.learning_utils import sigmoid, softmax, bce_loss, cross_entropy_loss, one_hot
from scipy.linalg import toeplitz


class MarkovChain:
    def __init__(self, n_states, sample_from_stationary_dist=False, init_dist=None):
        self.n_states = n_states
        self.sample_from_stationary_dist = sample_from_stationary_dist
        if not sample_from_stationary_dist:
            if init_dist is not None:
                self.init_dist = init_dist
            else:
                self.init_dist = self.default_init_dist()
            self.P = self._compute_transition_matrix()
            self.is_first_step = True    # flag for sampling first state from init_dist

    def _compute_transition_matrix(self):
        raise NotImplementedError

    def process_step(self):
        if self.sample_from_stationary_dist:
            self.s = sample_discrete(self.stationary_dist())
        else:
            if self.is_first_step:
                self.s = sample_discrete(self.init_dist)
                self.is_first_step = False
            else:
                self.s = sample_discrete(self.P[self.s, :])

    def default_init_dist(self):
        init_dist = np.zeros(self.n_states)
        init_dist[0] = 1.
        return init_dist

    @staticmethod
    def stationary_dist():
        raise NotImplementedError


class SimpleChain(MarkovChain):
    def __init__(self, epsilon, init_dist=None):
        self.set_epsilon(epsilon)
        super().__init__(n_states=2, init_dist=init_dist,
                         sample_from_stationary_dist=True if self.epsilon is None else False)

    def _compute_transition_matrix(self):
        return np.array([[1. - self.epsilon, self.epsilon],
                         [self.epsilon, 1. - self.epsilon]])

    def step(self):
        self.process_step()
        self.sample_data()

    def sample_data(self):
        raise NotImplementedError

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.P = self._compute_transition_matrix()

    @staticmethod
    def stationary_dist():
        return np.array([0.5, 0.5])


class CyclicChain(MarkovChain):
    def __init__(self, epsilon, n_states, init_dist=None):
        assert n_states > 2
        self.epsilon = epsilon
        super().__init__(n_states=n_states, init_dist=init_dist,
                         sample_from_stationary_dist=True if self.epsilon is None else False)

    def _compute_transition_matrix(self):
        toeplitz_col = np.zeros(self.n_states)
        toeplitz_col[0] = 1 - 2 * self.epsilon
        toeplitz_col[1] = self.epsilon
        toeplitz_col[-1] = self.epsilon
        return toeplitz(toeplitz_col)

    def step(self):
        self.process_step()
        self.sample_data()

    def sample_data(self):
        raise NotImplementedError

    def stationary_dist(self):
        return (1 / self.n_states) * np.ones(self.n_states)


class SimpleSwitchingChain(SimpleChain):
    def __init__(self, switch_rate, switch_values, init_dist=None):
        init_epsilon = np.min(switch_values)    # initial value is slowest mixing
        super().__init__(epsilon=init_epsilon, init_dist=init_dist)
        self.switch_rate = round(switch_rate)
        self.switch_values = switch_values
        self.n_transitions = 0      # counter for number of transitions -- for switching

    def process_step(self):
        super().process_step()
        self.n_transitions += 1
        if self.n_transitions % self.switch_rate == 0:
            new_epsilon = np.random.choice(self.switch_values)
            while new_epsilon == self.epsilon:
                new_epsilon = np.random.choice(self.switch_values)
            self.set_epsilon(new_epsilon)

    def sample_data(self):
        super().sample_data()


class SimpleChainLinearRegression(SimpleChain):
    def __init__(self, epsilon, n, dim, radius=None, sigma=0.1, init_dist=None):
        super().__init__(epsilon, init_dist)
        self.n = n if n % 2 == 0 else n+1   # n should be even
        self.dim = dim
        self.radius = radius
        self.sigma = sigma
        self._load_data()

    def _load_data(self):
        self._A = self.sigma * np.random.randn(self.n, self.dim)
        A1 = self._A[:int(self.n/2), :]
        A2 = self._A[int(self.n/2):, :]

        if self.radius is not None:
            # self._x1 = draw_in_l2_ball(dim=self.dim, R=self.radius)
            self._x1 = draw_in_l2_ball(dim=self.dim, R=self.radius/2)
            # self._x2 = draw_in_l2_ball(dim=self.dim, R=self.radius)
            self._x2 = draw_in_l2_ball(dim=self.dim, R=self.radius/2)
        else:
            self._x1 = np.random.randn(self.dim, 1)
            self._x2 = np.random.randn(self.dim, 1)
        # self._x2 = - self._x1.copy()
        # self._x2 = self._x1.copy()      # DEBUG -- equivalent of stationary
        b1 = A1 @ self._x1 + np.sqrt(0.001) * np.random.randn(int(self.n/2), 1)
        b2 = A2 @ self._x2 + np.sqrt(0.001) * np.random.randn(int(self.n/2), 1)
        self._b = np.concatenate([b1, b2], axis=0)

        self.data = {"0": (A1, b1), "1": (A2, b2)}

    def optimal_unconstrained_solution(self):
        return np.linalg.inv(self._A.T @ self._A) @ self._A.T @ self._b

    def sample_data(self):
        idx = np.random.randint(int(self.n/2))
        self.a = self.data[str(self.s)][0][idx].reshape(-1, 1)
        self.b = self.data[str(self.s)][1][idx]

    def grad(self, w, z=None):
        if z is not None:
            a, b = z[:-1].reshape(-1, 1), z[-1]
        else:
            a, b = self.a.copy(), self.b.copy()
        assert w.shape[0] == a.shape[0]
        return (a.T @ w - b) * a

    def evaluate(self, w):
        w_star = self.optimal_unconstrained_solution()
        loss_w = self.loss(w)
        suboptimality = loss_w - self.loss(w_star)
        return loss_w, suboptimality

    def loss(self, w):
        return (1 / (2 * self.n)) * np.linalg.norm(self._A @ w - self._b, axis=0) ** 2
        # return np.linalg.norm(self._A @ w - self._b, axis=0) ** 2

    def get_current_data(self):
        return np.concatenate([self.a, [self.b]]).reshape(-1, 1)


class SimpleSwitchingChainLinearRegression(SimpleSwitchingChain):
    def __init__(self, switch_rate, switch_values, n, dim,
                 radius=None, sigma=0.1, init_dist=None):
        super().__init__(switch_rate, switch_values, init_dist)
        self.n = n if n % 2 == 0 else n+1   # n should be even
        self.dim = dim
        self.radius = radius
        self.sigma = sigma
        self._load_data()

    def _load_data(self):
        self._A = self.sigma * np.random.randn(self.n, self.dim)
        A1 = self._A[:int(self.n/2), :]
        A2 = self._A[int(self.n/2):, :]

        if self.radius is not None:
            # self._x1 = draw_in_l2_ball(dim=self.dim, R=self.radius)
            self._x1 = draw_in_l2_ball(dim=self.dim, R=self.radius/2)
            # self._x2 = draw_in_l2_ball(dim=self.dim, R=self.radius)
            self._x2 = draw_in_l2_ball(dim=self.dim, R=self.radius/2)
        else:
            self._x1 = np.random.randn(self.dim, 1)
            self._x2 = np.random.randn(self.dim, 1)
        # self._x2 = - self._x1.copy()
        # self._x2 = self._x1.copy()      # DEBUG -- equivalent of stationary
        b1 = A1 @ self._x1 + np.sqrt(0.001) * np.random.randn(int(self.n/2), 1)
        b2 = A2 @ self._x2 + np.sqrt(0.001) * np.random.randn(int(self.n/2), 1)
        self._b = np.concatenate([b1, b2], axis=0)

        self.data = {"0": (A1, b1), "1": (A2, b2)}

    def optimal_unconstrained_solution(self):
        return np.linalg.inv(self._A.T @ self._A) @ self._A.T @ self._b

    def sample_data(self):
        idx = np.random.randint(int(self.n/2))
        self.a = self.data[str(self.s)][0][idx].reshape(-1, 1)
        self.b = self.data[str(self.s)][1][idx]

    def grad(self, w, z=None):
        if z is not None:
            a, b = z[:-1].reshape(-1, 1), z[-1]
        else:
            a, b = self.a.copy(), self.b.copy()
        assert w.shape[0] == a.shape[0]
        return (a.T @ w - b) * a

    def evaluate(self, w):
        w_star = self.optimal_unconstrained_solution()
        loss_w = self.loss(w)
        suboptimality = loss_w - self.loss(w_star)
        return loss_w, suboptimality

    def loss(self, w):
        return (1 / (2 * self.n)) * np.linalg.norm(self._A @ w - self._b, axis=0) ** 2
        # return np.linalg.norm(self._A @ w - self._b, axis=0) ** 2

    def get_current_data(self):
        return np.concatenate([self.a, [self.b]]).reshape(-1, 1)


class SimpleChainMNISTLogisticRegression(SimpleChain):
    def __init__(self, epsilon, digits=(3, 5), init_dist=None):
        super().__init__(epsilon, init_dist)
        self.digits = digits
        self._load_data()
        self.dim = self.test_X.shape[1] + 1     # dimension for optimization - +1 for bias

    def _load_data(self):
        (train_X_dig1, train_X_dig2), (test_X_dig1, test_X_dig2) = self.load_mnist_two_digits(self.digits)
        self.train_X = {"0": train_X_dig1, "1": train_X_dig2}
        self.test_X = np.concatenate((test_X_dig1, test_X_dig2), axis=0)
        self.test_y = np.concatenate((np.zeros(test_X_dig1.shape[0]), np.ones(test_X_dig2.shape[0]))).reshape(-1, 1)

    def sample_data(self):
        idx = np.random.randint(self.train_X[str(self.s)].shape[0])
        self.x = self.train_X[str(self.s)][idx]
        self.y = self.s

    def grad(self, w, z=None):
        if z is not None:
            x, y = z[:-1], z[-1]
        else:
            x, y = self.x.copy(), self.y.copy()
        assert w.shape[0] == x.shape[0] + 1
        x_tilde = np.concatenate([x, np.array([1.])]).reshape(-1, 1)
        logits = w.T @ x_tilde
        return (sigmoid(logits) - y) * x_tilde

    def evaluate(self, w):
        train_X = np.concatenate((self.train_X["0"], self.train_X["1"]), axis=0)
        train_y = np.concatenate((np.zeros((self.train_X["0"].shape[0], 1)),
                                  np.ones((self.train_X["1"].shape[0], 1))))
        train_X_tilde = np.concatenate((train_X, np.ones((train_X.shape[0], 1))), axis=1)
        test_X_tilde = np.concatenate((self.test_X, np.ones((self.test_X.shape[0], 1))), axis=1)
        train_y_pred, test_y_pred = sigmoid(train_X_tilde @ w), sigmoid(test_X_tilde @ w)

        train_loss, test_loss = bce_loss(train_y, train_y_pred), bce_loss(self.test_y, test_y_pred)
        train_acc, test_acc = np.mean((train_y_pred > 0.5).astype(int) == train_y, axis=0), \
                              np.mean((test_y_pred > 0.5).astype(int) == self.test_y, axis=0)
        return train_loss, train_acc, test_loss, test_acc

    def get_current_data(self):
        return np.concatenate([self.x, [self.y]]).reshape(-1, 1)

    @staticmethod
    def load_mnist_two_digits(digits):
        assert len(digits) == 2
        dig1, dig2 = digits[0], digits[1]

        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        dig1_train_loc, dig2_train_loc = np.where(train_y == dig1)[0], np.where(train_y == dig2)[0]
        dig1_test_loc, dig2_test_loc = np.where(test_y == dig1)[0], np.where(test_y == dig2)[0]

        train_X_dig1, train_X_dig2 = train_X[dig1_train_loc].reshape(-1, 784).astype(float) / 255., \
                                     train_X[dig2_train_loc].reshape(-1, 784).astype(float) / 255.
        test_X_dig1, test_X_dig2 = test_X[dig1_test_loc].reshape(-1, 784).astype(float) / 255., \
                                   test_X[dig2_test_loc].reshape(-1, 784).astype(float) / 255.

        return (train_X_dig1, train_X_dig2), (test_X_dig1, test_X_dig2)


class CyclicChainMNISTLogisticRegression(CyclicChain):
    def __init__(self, epsilon, init_dist=None):
        super().__init__(epsilon, n_states=10, init_dist=init_dist)
        self._load_data()
        self.dim = (self.test_X.shape[1] + 1) * 10  # dimension for optimization - +1 for bias, x10 since output is 10.

    def _load_data(self):
        train_data, test_data = self.load_mnist()
        self.train_X = {}
        for i, train_dig in enumerate(train_data):
            self.train_X[str(i)] = train_dig

        self.test_X = np.vstack(test_data)
        self.test_y = []
        for i, test_dig in enumerate(test_data):
            self.test_y.append(i * np.ones((test_dig.shape[0], 1)))
        self.test_y = np.vstack(self.test_y)

    def sample_data(self):
        idx = np.random.randint(self.train_X[str(self.s)].shape[0])
        self.x = self.train_X[str(self.s)][idx]
        self.y = int(self.s)

    def grad(self, w, z=None):
        if z is not None:
            x, y = z[:-1], z[-1]
        else:
            x, y = self.x.copy(), self.y
        W = np.reshape(w, (10, -1))
        x_tilde = np.concatenate([x, np.array([1.])]).reshape(-1, 1)
        logits = W @ x_tilde
        return np.outer(softmax(logits.T).T - one_hot(y, 10), x_tilde).reshape(-1, 1)

    def evaluate(self, w):
        '''
        :param w: size is 7850 x T
        :return:
        '''
        # train_X, train_y = [], []
        # for dig, train_dig in enumerate(self.train_X.values()):
        #     train_X.append(train_dig)
        #     train_y.append(dig * np.ones((train_dig.shape[0], 1)))
        # train_X = np.vstack(train_X)
        # train_y = np.vstack(train_y)

        # train_y_one_hot = one_hot(train_y[:, 0].astype(int), 10).reshape(-1, train_y.shape[0], 10)
        # train_y_one_hot = np.reshape(train_y_one_hot, -1, )
        test_y_one_hot = one_hot(self.test_y[:, 0].astype(int), 10).reshape(-1, self.test_y.shape[0], 10)

        # train_X_tilde = np.concatenate((train_X, np.ones((train_X.shape[0], 1))), axis=1)
        test_X_tilde = np.concatenate((self.test_X, np.ones((self.test_X.shape[0], 1))), axis=1)

        W_reshaped = w.T.reshape(-1, self.test_X.shape[-1] + 1, 10)
        # train_y_pred, test_y_pred = softmax(train_X_tilde @ W_reshaped), softmax(test_X_tilde @ W_reshaped)
        test_y_pred = softmax(test_X_tilde @ W_reshaped)
        # Loss
        # train_loss, test_loss = cross_entropy_loss(train_y_one_hot, train_y_pred), \
        #                         cross_entropy_loss(test_y_one_hot, test_y_pred)
        test_loss = cross_entropy_loss(test_y_one_hot, test_y_pred)
        # Accuracy
        # train_acc, test_acc = np.mean(np.argmax(train_y_pred, axis=-1) == train_y.T, axis=-1), \
        #                       np.mean(np.argmax(test_y_pred, axis=-1) == self.test_y.T, axis=-1)
        test_acc = np.mean(np.argmax(test_y_pred, axis=-1) == self.test_y.T, axis=-1)
        # return train_loss, train_acc, test_loss, test_acc
        return test_loss, test_acc

    def get_current_data(self):
        return np.concatenate([self.x, [self.y]]).reshape(-1, 1)

    @staticmethod
    def load_mnist():
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_data = []
        test_data = []
        for i in range(10):
            dig_i_loc_train = np.where(train_y == i)[0]
            dig_i_loc_test = np.where(test_y == i)[0]
            train_dig = train_X[dig_i_loc_train].reshape(-1, 784).astype(float) / 255.
            test_dig = test_X[dig_i_loc_test].reshape(-1, 784).astype(float) / 255.
            train_data.append(train_dig)
            test_data.append(test_dig)
        return train_data, test_data
