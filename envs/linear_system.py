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
        self.prev_X = self.X.copy()
        self.trajectory = [self.X]

    def get_dim(self):
        return self.A_star.shape[0]

    def get_curr_x(self):
        return self.X

    def step(self):
        self.prev_X = self.X.copy()
        self.X = np.matmul(self.A_star, self.X) + \
                 self.sigma * np.random.randn(self.d).reshape(-1, 1)
        self.trajectory.append(self.X)
        return self.X

    def grad(self, A, prev_X=None, X=None):  # TODO: replace self.prev_X and X with general X_t, X_{t+1} and if default is None use self.
        ''' OLS gradient at current time '''
        if (prev_X is None) or (X is None):
            prev_X = self.prev_X
            X = self.X
        return 2 * np.matmul(A @ prev_X - X, prev_X.T)


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
