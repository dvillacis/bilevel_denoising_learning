from abc import ABC, abstractmethod
import os
from numpy.core.shape_base import vstack
from pylops.basicoperators import FirstDerivative, HStack, Identity, VStack
from bdpl.Dataset import Dataset
from PIL import Image
import numpy as np
import pylops, pyproximal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def prodesc(u,v):
    prod=0
    dim = len(u)
    u = u.reshape((dim//2,2))
    v = v.reshape((dim//2,2))
    for i in range(dim//2):
        prod += np.dot(u[i,:],v[i,:])
    return prod

class LearningModel(ABC):
    def __init__(self,dataset:Dataset, λ_init=1.0, α_init=1.0) -> None:
        super().__init__()
        self.dataset = dataset
        self.λ = λ_init
        self.α = α_init
        self.den_list = None

    @abstractmethod
    def cost(self):
        raise NotImplementedError

    @abstractmethod
    def denoise(self):
        raise NotImplementedError

    @abstractmethod
    def gradient(self):
        raise NotImplementedError

    @abstractmethod
    def learn_data_parameter(self):
        raise NotImplementedError

    @abstractmethod
    def learn_reg_parameter(self):
        raise NotImplementedError

    def save(self,out_path):
        dendir = os.path.join(out_path,self.dataset.name)
        if not os.path.isdir(dendir):
            os.mkdir(dendir)
        summarydir = os.path.join(dendir,'summary.txt')
        with open(summarydir,'w') as f:
            f.write(f'Optimal_parameter: α: {np.linalg.norm(self.α)}, λ: {np.linalg.norm(self.λ)}\n')
            f.write(
                'num \t ssim_data \t ssim_rec \t psnr_data \t psnr_rec \t l2_data \t l2_rec \n')
            for i,den in enumerate(self.den_list):
                img,noisy = self.dataset.get_pair(i)
                f.write(f'{i:4} \t {ssim(img,noisy):.4f} \t {ssim(img,den):.4f} \t {psnr(img,noisy):.4f} \t {psnr(img,den):.4f} \t {np.linalg.norm(img.ravel()-noisy.ravel())**2:.4f} \t {np.linalg.norm(img.ravel()-den.ravel())**2:.4f}\n')
                im = Image.fromarray(den*255).convert("L")
                filedir = os.path.join(dendir,f'{self.dataset.name}_den_{i}.png')
                im.save(filedir,format='png')

class ScalarTVLearningModel(LearningModel):
    def __init__(self, dataset: Dataset, λ_init=1.0, α_init=1.0) -> None:
        super().__init__(dataset,λ_init,α_init)

    def cost(self):
        '''
        L2 cost associated with the recostructed dataset
        '''
        cost = 0
        for i,den in enumerate(self.den_list):
            img = self.dataset.get_original(i)
            cost += np.linalg.norm(img.ravel()-den.ravel())**2
        return 0.5*cost

    def denoise(self,niter=1000):
        self.den_list = []
        for pair in self.dataset.pairs:
            noisy = np.array(Image.open(pair[1]).convert("L"))# grab image as grayscale
            noisy = noisy / np.max(noisy) # normalization
            #print(f'Denoising {pair[1]} with parameters λ={self.λ} and α={self.α}...')
            Gop = pylops.Gradient(dims=self.dataset.img_size,kind='forward')
            l2 = pyproximal.L2(b=noisy.ravel(),sigma=self.λ)
            l21 = pyproximal.L21(ndim=2,sigma=self.α)
            L = 8
            tau = 1 / np.sqrt(L)
            mu = 1 / np.sqrt(L)
            den = pyproximal.optimization.primaldual.PrimalDual(l2,l21,Gop,tau=tau,mu=mu,x0=np.zeros_like(noisy.ravel()),niter=niter,theta=1.)
            self.den_list.append(den.reshape(noisy.shape))

    def gradient(self):
        '''
        Gradient calculation for the scalar problem
        '''
        img = self.dataset.get_original(0)
        den = self.den_list[0]
        dims = self.dataset.img_size
        ndims = np.prod(dims)
        Gop = pylops.Gradient(dims=dims,kind='forward')
        Kmx = FirstDerivative(ndims,dims=dims,dir=0,kind='backward')
        Kmy = FirstDerivative(ndims,dims=dims,dir=1,kind='backward')
        Kpx = FirstDerivative(ndims,dims=dims,dir=0,kind='forward')
        Kpy = FirstDerivative(ndims,dims=dims,dir=1,kind='forward')
        I = Identity(ndims)
        Z = pylops.Zero(ndims)
        A1 = HStack([self.λ*I,-Kmx,-Kmy])
        A2 = HStack([Kpx,I,Z])
        A3 = HStack([Kpy,Z,I])
        A = VStack([A1,A2,A3])
        b = np.hstack((den.ravel()-img.ravel(),np.zeros(2*ndims)))
        x0 = np.zeros(3*ndims)
        x = pylops.optimization.solver.cgls(A,b,x0)
        print(x)
        Kp = Gop(x[0][:ndims])
        Ku = Gop(den.ravel())
        print(-prodesc(Ku,Kp))

    def learn_data_parameter(self):
        pass

    def learn_reg_parameter(self):
        pass