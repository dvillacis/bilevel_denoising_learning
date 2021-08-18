import unittest
import numpy as np
import pylops
from bdpl.operators.ActiveOp import ActiveOp

np.random.seed(12345)

class ActiveOpTest(unittest.TestCase):
    def test_generation(self):
        u = np.random.rand(10,10)
        active = ActiveOp(u)
        print(active.active_set)
        x = np.ones(200)
        print(active(x))

    def test_concat(self):
        u = np.random.rand(10,10)
        Gop = pylops.Gradient(dims=u.shape,kind='forward')
        print(Gop(u.ravel()))
        Active = ActiveOp(u)
        print(Active*Gop(u.ravel()))