from unittest import TestCase

from backdoorpony.defence_helpers.autoencoder_util.AutoencoderCNN import AutoencoderCNN
import torch.optim as optim
import torch.nn as nn

class TestAutoencoderCNN(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutoencoderCNN, self).__init__(*args, **kwargs)
        self.input_shape = (1, 1, 1)
        self.path = 'path'
        self.cnn = AutoencoderCNN(self.input_shape, self.path)

    def test_init(self):
        self.assertTrue(isinstance(self.cnn, nn.Module))

    def test_get_opti(self):
        opti = self.cnn.get_opti()
        self.assertTrue(isinstance(opti, optim.Optimizer))

    def test_get_criterion(self):
        crit = self.cnn.get_criterion()
        self.assertTrue(isinstance(crit, nn.MSELoss))

    def test_get_nb_classes(self):
        nb_classes = self.cnn.get_nb_classes()
        self.assertTrue(nb_classes == self.input_shape[0] * self.input_shape[1]
                        * self.input_shape[2])

    def test_get_input_shape(self):
        input_shape = self.cnn.get_input_shape()
        self.assertTrue(input_shape == self.input_shape)

    def test_get_path(self):
        path = self.cnn.get_path()
        self.assertTrue(path == self.path + '_autoencoder')
