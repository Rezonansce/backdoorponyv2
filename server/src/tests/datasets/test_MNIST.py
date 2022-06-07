import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.models.image.MNIST import MNIST_CNN
from backdoorpony.classifiers.ImageClassifier import ImageClassifier

class TestDataLoader(TestCase):
    def test_get_data(self):
        with patch('backdoorpony.datasets.MNIST.load_mnist', return_value=((np.array([0, 1, 2]), np.array([0, 1, 2])), (np.array([3, 4, 5]), np.array([3, 4, 5])), 0, 5)):
            with patch('numpy.random.choice', return_value=np.array([1, 2])):
                mnist = MNIST(1)
                (x_raw_train, y_raw_train), (x_raw_test,
                                             y_raw_test), min, max = mnist.get_data()
                # Assert the get_data function works properly
                self.assertTrue(np.array_equal(x_raw_train, np.array([1, 2])))
                self.assertTrue(np.array_equal(y_raw_train, np.array([1, 2])))
                self.assertTrue(np.array_equal(
                    x_raw_test, np.array([3, 4, 5])))
                self.assertTrue(np.array_equal(
                    y_raw_test, np.array([3, 4, 5])))
                self.assertTrue(min == 0)
                self.assertTrue(max == 5)
                # Assert __init__ worked properly
                self.assertTrue(mnist.num_selection == 1)

if __name__ == '__main__':
    unittest.main()
