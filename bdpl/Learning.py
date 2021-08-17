from abc import ABC, abstractmethod
import os
from numpy.core.fromnumeric import ndim
from bdpl.Dataset import Dataset
from PIL import Image
import numpy as np
import pylops, pyproximal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class LearningModel(ABC):
    def __init__(self,dataset:Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    @abstractmethod
    def denoise(self):
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
    def __init__(self, dataset: Dataset, λ_init:int, α_init:int) -> None:
        super().__init__(dataset)
        self.λ = λ_init
        self.α = α_init
        self.den_list = None

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