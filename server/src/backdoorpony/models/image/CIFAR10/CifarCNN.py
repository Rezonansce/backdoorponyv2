import torch.nn as nn
import torch.optim as optim

__dataset__ = 'cifar10'
_class_name__ = 'CIFAR10_CNN'


class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.crit = nn.CrossEntropyLoss()
        self.nb_classes = 10
        self.path = 'cifar-10'
        self.input_shape = 3, 32, 32
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

    # TODO: Parameterize the methods, so we can do hyperparameter tuning

    def get_opti(self):
        '''
        Return the optimizer used for the CNN
        :return: stochastic gradient descent
        '''
        return optim.SGD(self.parameters(), lr=0.001)

    def get_criterion(self):
        '''
        Return the loss criterion used for the CNN
        :return: Cross-entropy loss
        '''
        return self.crit

    def get_nb_classes(self):
        '''
        Return the number of classes for the model
        :return: 10
        '''
        return self.nb_classes

    def get_input_shape(self):
        '''
        Return the shape of the input image
        :return: (3, 32, 32)
        '''
        return self.input_shape

    def get_path(self):
        '''
        Return the name of the pre-load state_dict of the model
        :return: cifar-10
        '''
        return self.path