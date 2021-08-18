import numpy as np
from pylops import LinearOperator, Gradient, FirstDerivative
import pylops
from pylops.utils.backend import get_array_module

def get_active(u,tol=1e-3):
    u = u.reshape(u.shape[0]//2,2)
    nu = np.sum(np.abs(u)**2,axis=-1)**(1./2)
    active = nu < tol
    inactive = nu >= tol
    return active

def get_inactive(u,tol=1e-3):
    u = u.reshape(u.shape[0]//2,2)
    nu = np.sum(np.abs(u)**2,axis=-1)**(1./2)
    inactive = nu >= tol
    return inactive

class ActiveOp(LinearOperator):
    def __init__(self,u,kind='centered',dtype=None):
        ndims = np.prod(u.shape)
        self.shape = (ndims,ndims)
        Gop = pylops.Gradient(u.shape)
        Ku = Gop(u.ravel())
        self.active = get_active(Ku)

    def _matvec(self,x):
        return self.active*x

    def _rmatvec(self,x):
        return self.active*x

class InactiveOp(LinearOperator):
    def __init__(self,u,kind='centered',dtype=None):
        ndims = np.prod(u.shape)
        self.shape = (ndims,ndims)
        Gop = pylops.Gradient(u.shape)
        Ku = Gop(u.ravel())
        self.inactive = get_inactive(Ku)

    def _matvec(self,x):
        return self.inactive*x

    def _rmatvec(self,x):
        return self.inactive*x


class NormOp(LinearOperator):
    def __init__(self,u,kind='centered',dtype=None):
        ndims = np.prod(u.shape)
        self.shape = (ndims,ndims)
        Gop = pylops.Gradient(u.shape)
        Ku = Gop(u.ravel())
        Ku = Ku.reshape(Ku.shape[0]//2,2)
        self.nKu = np.sum(np.abs(Ku)**2,axis=-1)**(1./2)

    def _matvec(self,x):
        return np.divide(x,self.nKu,out=np.zeros_like(x),where=self.nKu != 0)

    def _rmatvec(self,x):
        return self.nKu*x

class EpsOp(LinearOperator):
    def __init__(self, N, M=None, dtype='float64'):
        M = N if M is None else M
        self.shape = (N, M)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[0], dtype=self.dtype) + 1e-4

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[1], dtype=self.dtype) + 1e-4
        