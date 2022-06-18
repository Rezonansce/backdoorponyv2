'''
The below code generates a Convolutional Neural Network
for working with the MNIST dataset.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__dataset__ = 'mnist'
__class_name__ = 'MNIST_CNN'


class MNIST_CNN(nn.Module):
    def __init__(self):
        '''Initiates a CNN geared towards the MNIST dataset

        Returns
        ----------
        None
        '''
        super(MNIST_CNN, self).__init__()
        self.nb_classes = 10
        self.input_shape = 1, 28, 28
        self.crit = nn.CrossEntropyLoss()
        self.path = 'mnist'
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 50, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

    def get_opti(self):
        '''
        Get the optimizer used for MNIST.

        :return: Adam optimizer, with learning rate = 0.01
        '''
        return optim.Adam(self.parameters(), lr=0.01)

    def get_criterion(self):
        '''
        Get the loss criterion.

        :return: Cross-entropy loss
        '''
        return self.crit

    def get_nb_classes(self):
        '''
        Get the number of classes the model will have.

        :return:
        '''
        return self.nb_classes

    def get_input_shape(self):
        '''
        Get the shape of the input.
        First number is the number of channels of the image.

        :return: (1, 28, 28)
        '''
        return self.input_shape

    def get_path(self):
        '''
        Get the name of the pre-loaded model file.

        :return:
        '''
        return self.path
