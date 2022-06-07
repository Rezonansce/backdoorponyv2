import unittest
from unittest import TestCase
import numpy as np
from backdoorpony.datasets.CIFAR10 import CIFAR10


class TestDataLoader(TestCase):

    def test_data_format(self):
        '''
        Make sure the data comes in the right format
        '''
        cifar10 = CIFAR10(100)
        train_data, test_data = cifar10.get_datasets()
        # Assert __init__ worked properly
        self.assertTrue(cifar10.num_selection == 100)
        train_img = train_data[0]
        test_img = test_data[0]
        train_labels = train_data[1]
        test_labels = test_data[1]
        # Assert data shapes are correct
        self.assertTrue(np.shape(train_img) == (100, 3, 32, 32))
        self.assertTrue(np.shape(test_img) == (10000, 3, 32, 32))
        # Check that labels are between 0 and 9
        self.assertTrue((train_labels.min() >= 0.0) & (train_labels.max() <= 9.0))
        min_label = min(test_labels)
        max_label = max(test_labels)
        self.assertTrue((max_label <= 9) & (min_label >= 0))
        # Check that data is between 0 and 256
        self.assertTrue((train_img.max() <= 256) & (train_img.min() >= 0))
        self.assertTrue((test_img.max() <= 256) & (test_img.min() >= 0))

if __name__ == '__main__':
    unittest.main()