
import numpy as np
from numpy import random
import numpy.linalg as la


def draw_in_l2_ball(dim, R, size=1):
    assert isinstance(size, int)
    if size == 1:
        dir = random.normal(size=dim)
        dir /= la.norm(dir)
        radius = random.random() ** (1/dim)
    else:
        dir = random.normal(size=(size, dim))
        dir /= la.norm(dir, axis=1, keepdims=True)
        radius = random.rand(size, 1) ** (1/dim)
    return R * dir * radius


def project_l2_ball(x, R):
    if R is None:
        return x
    if la.norm(x) < R:
        return x
    else:
        return R * x / la.norm(x)
