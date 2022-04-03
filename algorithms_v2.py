import numpy as np
from numpy.linalg import norm
from utils.projection_utils import project_l2_ball
from utils.prob_utils import sample_discrete
from learning_rate_schedulers import *
from utils.Buffer import Buffer


class SGD:
    def __init__(self, w_init, lr_params, momentum_def=None, beta=0.9, grad_clip=None):
        self.w = w_init  # reshape([-1 1])
        self.iterates = [self.w]
        self.lr_params = lr_params
        self.set_lr_scheduler()
        self.momentum_def = momentum_def
        self.beta = beta  # Momentum parameter
        if self.momentum_def is not None:
            self.m = None
        self.grad_clip = grad_clip

    def step(self, t, objective, project):
        gt = self.compute_grad(objective, self.w)
        if self.grad_clip is not None:
            gt = np.clip(gt, -self.grad_clip, self.grad_clip)
        # lr = self.compute_lr(t, gt, self.T, objective)
        lr = self.compute_lr(t, gt)
        if self.momentum_def is not None:
            if self.m is None:
                self.m = gt
            else:
                if self.momentum_def == 'standard':
                    self.m = self.beta * self.m + (1 - self.beta) * gt
                elif self.momentum_def == 'corrected':
                    self.m = self.beta * self.m + (1 - self.beta) * gt + \
                             self.beta * (gt - self.compute_grad(objective, self.iterates[-2]))
                else:
                    raise NotImplementedError
            if project:
                assert hasattr(objective, 'radius')
                self.w = project_l2_ball(self.w - lr * self.m, R=objective.radius)
            else:
                self.w = self.w - lr * self.m
        else:
            if project:
                assert hasattr(objective, 'radius')
                self.w = project_l2_ball(self.w - lr * gt, R=objective.radius)
            else:
                self.w = self.w - lr * gt

    def run(self, T, objective, project=False):
        for t in range(1, T + 1):
            objective.step()
            self.step(t, objective, project)
            self.iterates.append(self.w)

    def compute_lr(self, t, grad):
        # return self.lr_scheduler.compute_lr(t=t, grad=grad, objective=objective)
        return self.lr_scheduler.compute_lr(t=t, grad=grad)

    def compute_grad(self, objective, w=None):
        if w is None:
            w = self.w.copy()
        return objective.grad(w)

    def set_lr_scheduler(self):
        if self.lr_params["type"] == 'alpha/sqrt(t)':
            self.lr_scheduler = OneOverSqrtScheduler(self.lr_params["alpha"])
        elif self.lr_params["type"] == 'const':
            self.lr_scheduler = ConstantStepSizeScheduler(self.lr_params["alpha"])
        elif self.lr_params["type"] == 'AdaGrad':
            self.lr_scheduler = AdaGradScheduler(self.lr_params["alpha"])
        elif self.lr_params["type"] == '1/2R':
            self.lr_scheduler = ConstantStepSizeScheduler(1 / (2 * self.lr_params["R"]))
        else:
            raise NotImplementedError


class MiniBatchSGD(SGD):
    def __init__(self, w_init, lr_params, momentum_def=None, beta=0.9, grad_clip=None, batch_size=1):
        super().__init__(w_init, lr_params, momentum_def, beta, grad_clip)
        self.batch_size = batch_size
        self.batch_samples = []

    def run(self, T, objective, project=False):
        t = 1
        while t <= T:
            self.draw_batch_samples(objective)
            self.step(t, objective, project)
            self.iterates.append(self.w)
            self.batch_samples = []  # reset saved batch samples.
            t += self.batch_size

    def draw_batch_samples(self, objective):
        for i in range(self.batch_size):
            objective.step()
            self.batch_samples.append(objective.get_current_data())
        self.batch_samples = np.hstack(self.batch_samples)

    def compute_grad(self, objective, w=None):
        if w is None:
            w = self.w.copy()
        grads = []
        for i in range(self.batch_size):
            grads.append(objective.grad(w, self.batch_samples[:, i]))
        grads = np.hstack(grads)
        return np.mean(grads, axis=1, keepdims=True)


class SGD_MLMC(SGD):
    def __init__(self, w_init, lr_params, truncate_at=None, momentum_def=None,
                 beta=0.9, grad_clip=None):
        super().__init__(w_init, lr_params, momentum_def, beta, grad_clip)
        self.geometric_param = 0.5  # const at MAG
        self.truncate_at = truncate_at
        if self.truncate_at is not None:
            normalization_factor = (self.geometric_param / (1 - self.geometric_param)) * \
                                   (1 / (1 - ((1 - self.geometric_param) ** self.truncate_at)))
            self.truncated_dist = normalization_factor * \
                                  (1 - self.geometric_param) ** np.arange(1, self.truncate_at + 1)
        self.n_samples_per_step_list = [0]  # list for saving the number of samples
        self.samples_for_mlmc = []  # list for saving process samples at a single step

    def run(self, T, objective, project=False):
        t = 1
        while t <= T:
            self.draw_samples_for_mlmc(objective, T)
            # self.draw_samples_for_mlmc(objective, t)
            self.n_samples_per_step_list.append(self.samples_for_mlmc.shape[-1])
            self.step(t, objective, project)
            self.iterates.append(self.w)
            self.samples_for_mlmc = []  # reset saved MLMC samples for next step!
            t += self.n_samples_per_step_list[-1]

    def draw_samples_for_mlmc(self, objective, T):
        if self.truncate_at is not None:
            Jt = sample_discrete(self.truncated_dist) + 1
        else:
            Jt = np.random.geometric(p=self.geometric_param)
        objective.step()
        self.samples_for_mlmc.append(objective.get_current_data())
        Nt = 2 ** Jt  # Number of samples from the process
        if Nt <= T:
            for j in range(Nt - 1):
                objective.step()
                self.samples_for_mlmc.append(objective.get_current_data())
        self.samples_for_mlmc = np.hstack(self.samples_for_mlmc)

    def compute_grad(self, objective, w=None):
        if w is None:
            w = self.w.copy()
        Nt = self.samples_for_mlmc.shape[-1]
        grads = []
        for j in range(Nt):
            grads.append(objective.grad(w, self.samples_for_mlmc[:, j]))
        if Nt == 1:
            grad = grads[0]
        else:
            grads = np.hstack(grads)
            grad = grads[:, 0].reshape(-1, 1) + Nt * (np.mean(grads, axis=1, keepdims=True) -
                                                      np.mean(grads[:, :int(Nt / 2)], axis=1, keepdims=True))
        return grad


class SGD_DD(SGD):
    def __init__(self, w_init, lr_params, drop_param, momentum_def=None, beta=0.9, grad_clip=None):
        super().__init__(w_init, lr_params, momentum_def, beta, grad_clip)
        assert isinstance(drop_param, int)
        self.drop_param = drop_param

    def run(self, T, objective, project=False):
        for t in range(1, T + 1):
            objective.step()
            if t % self.drop_param == 0:
                self.step(t, objective, project)
                self.iterates.append(self.w)


class SGD_ER(SGD):
    def __init__(self, w_init, lr_params, momentum_def=None, beta=0.9, grad_clip=None):
        super().__init__(w_init, lr_params, momentum_def, beta, grad_clip)
        self.buff = Buffer()

    def run(self, T, objective, project=False):
        for t in range(1, T + 1):
            if self.buff.len() == 0:
                self.buff.store(objective.get_curr_x())
            if t % 100_000 == 0:
                print(t, " iterations passed !")
            self.buff.store(objective.step())
            self.step(t, objective, project)
            self.iterates.append(self.w)

    def compute_grad(self, objective, w=None):
        if w is None:
            w = self.w.copy()
        Z_t = self.buff.sample()
        return objective.grad(w, Z_t[0], Z_t[1])

