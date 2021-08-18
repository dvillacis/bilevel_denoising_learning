import numpy as np
from pylops import LinearOperator, Gradient

def get_active(u):
    u = u.reshape(u.shape[0]//2,2)
    nu = np.sum(np.abs(u)**2,axis=-1)**(1./2)
    nu = nu < 1e-3
    return np.hstack((nu,nu))

class ActiveOp(LinearOperator):
    def __init__(self,u,kind='centered',dtype=None):
        dims = u.shape
        ndims = np.prod(dims)
        self.shape = (2*ndims,2*ndims)
        self.dtype = np.dtype(dtype)
        Gop = Gradient(dims=dims,kind=kind)
        Gu = Gop(u.ravel())
        self.active_set = get_active(Gu)

    def _matvec(self,x):
        return self.active_set*x

    def _rmatvec(self,x):
        return self.active_set*x
