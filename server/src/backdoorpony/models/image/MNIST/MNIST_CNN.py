'''
The below code generates a Convolutional Neural Network
for working with the MNIST dataset.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__name__ = "MNIST_CNN"
__category__ = 'image'
__input_type__ = "image"
__defaults__ = {
    'learning_rate': {
        'pretty_name': 'Learning Rate',
        'default_value': [0.01],
        'info': 'The learning rate of the optimizer'
    },
    'optim': {
        'pretty_name': 'Optimizer',
        'default_value': ['Adam'],
        'info': 'The optimizer used in the training process. Currently, only "Adam" and "SGD" are available.' +
                'If the input is not valid, Adam optimizer will be chosen.'
    },
    'pre_load': {
        'pretty_name': 'Preload Model',
        'default_value': ['False'],
        'info': 'True if you would like to use a pre-trained model with default hyperparameters. False otherwise.'
    },
    'num_selection': {
        'pretty_name': 'Number of samples',
        'default_value': [50000],
        'info': 'The number of samples used to train the model. Max 50000.'
    }
}
__link__ = None,
__info__ = '''A model that trains image input'''
# __dataset__ = 'mnist'
# __class_name__ = 'MNIST_CNN'


class MNIST_CNN(nn.Module):

    def __init__(self, model_parameters):
        '''
        Initiate a MNIST CNN

        :param model_parameters: hyperparameters for the model
        '''
        super(MNIST_CNN, self).__init__()
        # If do_preload is true, use the default parameters of the model
        self.do_preload = model_parameters['pre_load']['value'][0].lower()
        if self.do_preload == 'true':
            self.do_preload = True
        else:
            self.do_preload = False
        self.lr = model_parameters['learning_rate']['value'][0]
        if model_parameters['optim']['value'][0] == 'SGD':
            self.optim = 'SGD'
        else:
            self.optim = 'Adam'
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
        if self.optim == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        else:
            return optim.SGD(self.parameters(), lr=self.lr)

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

    def get_do_pre_load(self):
        '''
        Return True if the model should use a pre-load
        Return False otherwise
        :return:
        '''
        return self.do_preload