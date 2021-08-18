import os
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = None
        self.num_samples = None
        self.img_size = None
        self.pairs = []

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_pair(self,i):
        pass

    def __repr__(self) -> str:
        return f'<Dataset name:{self.name},num_samples:{self.num_samples},img_size:{self.img_size}>'

    def __str__(self) -> str:
        return f'Dataset name:{self.name}, num_samples:{self.num_samples}, img_size:{self.img_size}'

class DirDataset(Dataset):
    def load(self,path):
        if os.path.isdir(path):
            filelist = os.path.join(path,'filelist.txt')
            if os.path.isfile(filelist):
                self.name = os.path.basename(path)
                with open(filelist,'r') as reader:
                    sz = None
                    for line in reader.readlines():
                        t = line.strip().split(',')
                        t0 = os.path.join(path,t[0])
                        t1 = os.path.join(path,t[1])
                        t0_img = Image.open(t0)
                        t1_img = Image.open(t0)
                        if t0_img.size != t1_img.size:
                            raise ValueError('Size in image pair differs, exiting...')
                        sz = t0_img.size
                        t0_img.close()
                        t1_img.close()
                        self.pairs.append((t0,t1))
                self.img_size = sz
                self.num_samples = len(self.pairs)
            else:
                raise ValueError(f'File {filelist} not found, corrupted dataset...')
        else:
            raise ValueError(f'Directory {path} does not exist...')

    def get_pair(self,i):
        if i > self.num_samples:
            raise ValueError(f'Dataset {self.name} only has {self.num_samples} samples...')
        img = np.array(Image.open(self.pairs[i][0]).convert("L"))
        img = img / np.max(img)
        noisy = np.array(Image.open(self.pairs[i][1]).convert("L"))
        noisy = noisy / np.max(noisy)
        return img,noisy

    def get_original(self,i):
        if i > self.num_samples:
            raise ValueError(f'Dataset {self.name} only has {self.num_samples} samples...')
        img = np.array(Image.open(self.pairs[i][0]).convert("L"))
        img = img / np.max(img)
        return img

    def get_noisy(self,i):
        if i > self.num_samples:
            raise ValueError(f'Dataset {self.name} only has {self.num_samples} samples...')
        noisy = np.array(Image.open(self.pairs[i][1]).convert("L"))
        noisy = noisy / np.max(noisy)
        return noisy

#TODO: Add dummy dataset with the black square