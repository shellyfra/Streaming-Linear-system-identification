import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LinearSystem:
    '''
    Continuous linear system with i.i.d. Normal noise terms.
    '''

    def __init__(self, A_star, sigma, X0=None):
        self.A_star = A_star
        self.sigma = sigma
        self.d = self.get_dim()
        if X0 is None:
            X0 = np.zeros((self.d, 1))
        self.X0 = X0
        # Init. sequence
        self.X = self.X0.copy()
        self.A = np.zeros((self.d, self.d))
        self.prev_X = self.X0.copy()
        self.points = [self.X]

    def get_dim(self):
        return self.A_star.shape[0]

    def step(self):
        self.prev_X = self.X.copy()
        self.X = np.matmul(self.A_star, self.X) + \
                 self.sigma * np.random.randn(self.d).reshape(-1, 1)
        self.points.append(self.X)

    def grad(self, w):
        return 2 * np.matmul(self.A @ self.prev_X - self.X, self.prev_X.T)

class RandBiMod(LinearSystem):
    def __init__(self, d=5, rho=0.9, sigma=1, X0=None):
        self.d = d
        self.rho = rho
        super().__init__(A_star=self.compute_A_star(), sigma=sigma, X0=X0)

    def compute_A_star(self):
        # eigenvalues
        upper_diag = self.rho * np.ones(np.ceil(self.d / 2).astype(int))
        lower_diag = (self.rho / 3) * np.ones(self.d - upper_diag.shape[0])
        Lambda = np.diag(np.concatenate((upper_diag, lower_diag)))
        # eigenvectors
        U = ortho_group.rvs(self.d)
        return U @ Lambda @ U.T
