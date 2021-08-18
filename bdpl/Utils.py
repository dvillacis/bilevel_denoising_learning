
import numpy as np

def prodesc(u,v):
    prod=0
    dim = len(u)
    u = u.reshape((dim//2,2))
    v = v.reshape((dim//2,2))
    print(u[0,:].shape)
    for i in range(dim//2):
        prod += np.dot(u[i,:],v[i,:])