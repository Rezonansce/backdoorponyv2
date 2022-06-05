import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__dataset__ = 'fashionmnist'
__class_name__ = 'FMNIST_CNN'

class FMNIST_CNN(nn.Module):

    def __init__(self):
        super(FMNIST_CNN, self).__init__()
        self.crit = nn.CrossEntropyLoss()
        self.path = 'fashion_mnist'
        self.nb_classes = 10
        self.input_shape = 1, 28, 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

    def get_opti(self):
        '''
        Get the optimizer used for MNIST.

        :return: Adam optimizer, with learning rate = 0.01
        '''
        return optim.Adam(self.parameters(), lr=0.001)

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