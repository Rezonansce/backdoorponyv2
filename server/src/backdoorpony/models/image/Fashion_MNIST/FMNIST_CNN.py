import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__name__ = "FMNIST_CNN"
__category__ = 'image'
__input_type__ = "image"
__defaults__ = {
    'learning_rate': {
        'pretty_name': 'Learning rate',
        'default_value': [0.001],
        'info': 'Learning rate used for the training of the model.'
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
__link__ = 'Why would there be a link to the model page???'
__info__ = '''A model that trains image input'''
# __dataset__ = 'fashionmnist'
# __class_name__ = 'FMNIST_CNN'

class FMNIST_CNN(nn.Module):

    def __init__(self, model_parameters):
        super(FMNIST_CNN, self).__init__()
        self.do_preload = model_parameters['pre_load']['value'][0]
        if self.do_preload == 'True':
            self.do_preload = True
            self.optim = 'Adam'
            self.lr = 0.01
        else:
            self.do_preload = False
            self.lr = model_parameters['learning_rate']['value'][0]
            if model_parameters['optim']['value'][0] == 'SGD':
                self.optim = 'SGD'
            else:
                self.optim = 'Adam'
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