# -*- coding: utf-8 -*-
"""
Created on Fri May  6 02:35:54 2022

@author: kikig
"""

import torch
import torch.nn as nn


class Audio_MNIST_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''Initiates a RNN geared towards the IMDB dataset

        Returns
        ----------
        None
        '''
        super(Audio_MNIST_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.i2o = nn.Linear(input_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        combined = input#torch.cat((input, hidden), 1)
        #hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

