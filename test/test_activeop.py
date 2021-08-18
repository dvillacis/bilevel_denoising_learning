import unittest
import numpy as np
import pylops
from bdpl.operators.IndexOp import ActiveOp,InactiveOp

np.random.seed(12345)

class IndexOpTest(unittest.TestCase):
    def test_generation(self):
        u = np.random.rand(10,10)
        Act = ActiveOp(u)
        Inact = InactiveOp(u)
        x = np.random.rand(100)
        print(Act(x))
        print(Inact(x))

    # def test_concat(self):
    #     u = np.random.rand(10,10)
    #     Gop = pylops.Gradient(dims=u.shape,kind='forward')
    #     print(Gop(u.ravel()))
    #     Active = ActiveOp(u)
    #     print(Active*Gop(u.ravel()))