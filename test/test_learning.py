import unittest
import warnings
from bdpl.Dataset import DirDataset
from bdpl.Learning import ScalarTVLearningModel

class LearningTest(unittest.TestCase):
    # def test_scalar_tv_denoising(self):
    #     ds = DirDataset()
    #     ds.load('datasets/cameraman_128_5')
    #     #ds.load('datasets/faces_train_128_10')
    #     model = ScalarTVLearningModel(ds,35.0,1.0)
    #     model.denoise()
    #     model.save('output')
    #     c = model.cost()
    #     print(c)

    # def test_gradient_scalar_tv_denoising(self):
    #     warnings.filterwarnings("ignore",category=DeprecationWarning)
    #     ds = DirDataset()
    #     ds.load('datasets/cameraman_128_5')
    #     #ds.load('datasets/faces_train_128_10')
    #     model = ScalarTVLearningModel(ds,λ_init=90.0)
    #     model.denoise()
    #     model.gradient()

    def test_learning_scalar_tv_denoising(self):
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        ds = DirDataset()
        ds.load('datasets/cameraman_128_5')
        #ds.load('datasets/faces_train_128_10')
        model = ScalarTVLearningModel(ds,λ_init=70.0)
        model.learn_data_parameter()