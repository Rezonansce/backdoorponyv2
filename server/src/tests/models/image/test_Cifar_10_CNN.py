import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch.optim as optim
import torch.nn as nn

from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN

class TestCifarCNN(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCifarCNN, self).__init__(*args, **kwargs)
        self.model_params = {'learning_rate': {'value': [0.001]},
                             'optim': {'value': ['SGD']},
                             'pre_load': {'value': ["True"]},
                             'num_selection': {'value': [1234]}}
        self.cnn = CifarCNN(model_parameters=self.model_params)

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

    def test_get_do_pre_load(self):
        self.assertTrue(self.cnn.get_do_pre_load())