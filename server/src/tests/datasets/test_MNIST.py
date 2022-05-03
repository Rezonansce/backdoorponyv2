import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from backdoorpony.datasets.MNIST import MNIST


class TestDataLoader(TestCase):
    def test_get_data(self):
        # Use the function from where it is called.
        # Since MNIST.py has 'from art.utils import load_mnist' so we patch MNIST.load_mnist instead of art.utils.load_mnist
        with patch('backdoorpony.datasets.MNIST.load_mnist', return_value=((np.array([0, 1, 2]), np.array([0, 1, 2])), (np.array([3, 4, 5]), np.array([3, 4, 5])), 0, 5)):
            # Since we call np.random.choice instead of just choice we do need the full definition
            # As np is an alias we need to patch the original name so numpy.random.choice instead of np.random.choice
            # If we had 'from numpy.random import choice' instead of 'import numpy as np' we would've had to patch MNIST.choice
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
