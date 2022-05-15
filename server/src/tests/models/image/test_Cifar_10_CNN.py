import unittest
from unittest import TestCase

import numpy as np
import torch.optim as optim
import torch.nn as nn

from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN

class TestDataLoader(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.cnn = CifarCNN()

    def test_get_opti(self):
        optimizer = self.cnn.get_opti()
        self.assertTrue(isinstance(optimizer, optim.Optimizer))

    def test_get_criterion(self):
        criterion = self.cnn.get_criterion()
        self.assertTrue(isinstance(criterion, nn.CrossEntropyLoss))

    def test_get_nb_classes(self):
        nb_classes = self.cnn.get_nb_classes()
        self.assertTrue(nb_classes == 10)

    def test_get_input_shape(self):
        input_shape = self.cnn.get_input_shape()
        self.assertTrue(input_shape == (3, 32, 32))

    def test_get_path(self):
        path = self.cnn.get_path()
        self.assertTrue(path == 'cifar-10')