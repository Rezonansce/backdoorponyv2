import unittest
from unittest import TestCase

from backdoorpony.models.image.Fashion_MNIST.FMNIST_CNN import FMNIST_CNN
import torch.optim as optim
import torch.nn as nn

class TestFmnistCNN(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFmnistCNN, self).__init__(*args, **kwargs)
        self.model_params = {'learning_rate': {'value': [0.001]},
                             'optim': {'value': ['SGD']},
                             'pre_load': {'value': ["True"]},
                             'num_selection': {'value': [1234]}}
        self.cnn = FMNIST_CNN(self.model_params)


    def test_init(self):
        self.assertTrue(isinstance(self.cnn, nn.Module))

    def test_get_opti(self):
        opti = self.cnn.get_opti()
        self.assertTrue(isinstance(opti, optim.Optimizer))

    def test_get_criterion(self):
        crit = self.cnn.get_criterion()
        self.assertTrue(isinstance(crit, nn.CrossEntropyLoss))

    def test_get_nb_classes(self):
        nb_classes = self.cnn.get_nb_classes()
        self.assertTrue(nb_classes == 10)

    def test_get_input_shape(self):
        input_shape = self.cnn.get_input_shape()
        self.assertTrue(input_shape == (1, 28, 28))

    def test_get_path(self):
        path = self.cnn.get_path()
        self.assertTrue(path == 'fashion_mnist')
