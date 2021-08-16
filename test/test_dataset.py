import unittest

from bdpl.Dataset import DirDataset

class DirDatasetTest(unittest.TestCase):
    def test_dataset_load(self):
        ds = DirDataset()
        ds.load('datasets/cameraman_128_5')
        print(ds)