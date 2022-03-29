import numpy as np
from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute_lr(self, *args):
        pass


class ConstantStepSizeScheduler(LearningRateScheduler):
    def __init__(self, alpha=1):
        super().__init__()
        self.learning_rates = []
        self.alpha = alpha

    def compute_lr(self, **kwargs):
        learning_rate = self.alpha
        self.learning_rates.append(learning_rate)
        return learning_rate


class OneOverSqrtScheduler(LearningRateScheduler):
    def __init__(self, alpha=1):
        super().__init__()
        self.learning_rates = []
        self.alpha = alpha

    def compute_lr(self, **kwargs):
        t = kwargs['t']
        assert t != 0
        learning_rate = self.alpha / np.sqrt(t)
        self.learning_rates.append(learning_rate)
        return learning_rate


class AdaGradScheduler(LearningRateScheduler):
    def __init__(self, alpha=1, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.G_sqr = self.epsilon  # Initialization
        self.learning_rates = []

    def compute_lr(self, **kwargs):
        '''
        :param args: args[1] is gradient (numpy array)
        '''
        self.G_sqr += (np.linalg.norm(kwargs['grad']) ** 2)
        learning_rate = self.alpha / np.sqrt(self.G_sqr)
        self.learning_rates.append(learning_rate)
        return learning_rate


# class LinearSystemScheduler(LearningRateScheduler):
#     def __init__(self, samples_num=10_000_000):
#         super().__init__()
#         self.learning_rate = None
#         self.T = samples_num
#
#     def compute_lr(self, **kwargs):
#         """
#         To estimate R from the data, we use the first floor(2logT) = 32 samples and set
#         R as the sum of the norms of these samples. we let the step size to be 1/2R
#         """
#         if self.learning_rate is not None:
#             return self.learning_rate
#
#         T = kwargs['T']
#         objective = kwargs['objective']
#         R_num = int(np.floor(2*np.log(T)))
#         R = 0
#         for _ in range(R_num):
#             objective.step()
#             R += np.linalg.norm(objective.X)
#         self.learning_rate = 1/(2*R)
#         objective.points = [objective.prev_X, objective.X]  # reset points count
#         return self.learning_rate

