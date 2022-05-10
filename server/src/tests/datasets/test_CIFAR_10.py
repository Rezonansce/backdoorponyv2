import unittest
from unittest import TestCase


import numpy as np
from backdoorpony.datasets.CIFAR10 import CIFAR10

class TestDataLoader(TestCase):
    def test_get_data(self):
        cifar = CIFAR10(5)
        (x_train, y_train), (x_test, y_test) = cifar.get_datasets()
        self.assertTrue(len(x_train) == 5)
        self.assertTrue(np.shape(x_train) == (5, 3, 32, 32))
        self.assertTrue(np.shape(x_test) == (10000, 3, 32, 32))
        self.assertTrue(np.shape(y_train) == (5,))
        self.assertTrue(np.shape(y_test) == (10000,))


if __name__ == '__main__':
    unittest.main()