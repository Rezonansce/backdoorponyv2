import unittest
import numpy as np

from backdoorpony.datasets.Fashion_MNIST import Fashion_MNIST


class TestFmnist(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFmnist, self).__init__(*args, **kwargs)
        self.size = 500
        self.dataloader = Fashion_MNIST(self.size)

    def test_init(self):
        self.assertTrue(self.dataloader)
        self.assertTrue(self.dataloader.num_selection == self.size)

    def test_get_data(self):
        # Checks the format of the data
        train_data, test_data = self.dataloader.get_datasets()
        train_imgs = train_data[0]
        test_imgs = test_data[0]
        train_labels = train_data[1]
        test_labels = test_data[1]
        # Check shapes
        self.assertTrue(np.shape(train_imgs) == (self.size, 1, 28, 28))
        self.assertTrue(np.shape(test_imgs) == (10000, 1, 28, 28))
        self.assertTrue(np.shape(test_labels) == (10000,))


