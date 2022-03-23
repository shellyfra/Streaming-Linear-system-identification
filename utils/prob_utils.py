
import numpy as np
from numpy.linalg import norm

def sample_discrete(probs):
    """
    Samples a discrete distribution
     Parameters:
        probs - probabilities over {0,...K-1}
    Returns:
        drawn random integer from  {0,...K-1} according to probs
    """
    K = probs.size
    return np.random.choice(K, size=1, p=probs)[0]


def stationary_distribution_of_markov_chain(P):
    '''
    Computes stationary distribution of a Markov chain.
    from: https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
    :param P: transition matrix of Markov chain [nS]x[nS]
    :return:
    '''
    evals, evecs = np.linalg.eig(P.T)
    evec1 = evecs[:, np.isclose(evals, 1)]

    # Since np.isclose will return an array, we've indexed with an array
    # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    # eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    return stationary.real


def draw_in_l2_ball(dim, R, size=1):
    random_dir = np.random.normal(size=(dim, size))
    random_dir /= norm(random_dir, axis=0)
    random_radius = np.random.random(size) ** (1 / dim)
    return R * random_dir * random_radius
