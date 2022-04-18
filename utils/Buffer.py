import random
from collections import deque
from random import randint
import numpy as np


class Buffer:
    """
    Buffer class for the buffer in the SGD-ER and SGD-RER
    """
    def __init__(self, max_length=1_000_000):
        self.buff = list()
        self.max_length = max_length
        self.need_del = False

    def len(self):
        return len(self.buff)

    def store(self, x):
        self.buff.append(x)
        if self.need_del or (self.len() > self.max_length):
            self.buff.pop(0)
            self.need_del = True
            # del self.buff[0]

    def sample(self):
        index = randint(0, len(self.buff)-2)
        z_t = (self.buff[index], self.buff[index + 1])
        return z_t

    def get_item_from_index(self, index):
        return self.buff[index]
