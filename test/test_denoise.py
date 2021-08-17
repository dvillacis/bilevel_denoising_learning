import unittest

from bdpl.Dataset import DirDataset
from bdpl.Learning import ScalarTVLearningModel

class DenoiseTest(unittest.TestCase):
    def test_scalar_tv_denoising(self):
        ds = DirDataset()
        ds.load('datasets/cameraman_128_5')
        model = ScalarTVLearningModel(ds,35.0,1.0)
        model.denoise()
        model.save('output')
        