import random
from collections import deque
from random import randint
import numpy as np


class Buffer:
    """
    Buffer class for the buffer in the SGD-ER and SGD-RER
    """
    def __init__(self):
        self.buff = deque()

    def len(self):
        return len(self.buff)

    def store(self, x):
        self.buff.append(x)

    def sample(self):
        index = randint(0, len(self.buff)-2)  # minus the last index
        i = round(random.uniform(0,1)*(len(self.buff)-2))
        z_t = (self.buff[i], self.buff[i + 1])
        return z_t
