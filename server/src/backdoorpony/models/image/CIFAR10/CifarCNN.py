import torch.nn as nn
import torch.optim as optim

__name__ = "CifarCNN"
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
        'default_value': ['SGD'],
        'info': 'The optimizer used in the training process. Currently, only "Adam" and "SGD" are available.' +
                'If the input is not valid, SGD optimizer will be chosen.'
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
__info__ = '''A model that trains text input'''
# __dataset__ = 'cifar10'
# _class_name__ = 'CIFAR10_CNN'


class CifarCNN(nn.Module):

    def __init__(self, model_parameters):
        '''
        Initiates a CNN for the Cifar-10 dataset

        :param model_parameters: Hyperparameters for the model
        '''
        super(CifarCNN, self).__init__()
        # If do_preload is true, use the default parameters of the model
        self.do_preload = model_parameters['pre_load']['value'][0].lower()
        if self.do_preload == 'true':
            self.do_preload = True
        else:
            self.do_preload = False
        self.lr = model_parameters['learning_rate']['value'][0]
        if model_parameters['optim']['value'][0] == 'Adam':
            self.optim = 'Adam'
        else:
            self.optim = 'SGD'
        self.crit = nn.CrossEntropyLoss()
        self.nb_classes = 10
        self.path = 'cifar-cnn' + '_' + self.optim + '_' + str(self.lr) + '.pt'
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
        if self.optim == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        else:
            return optim.SGD(self.parameters(), lr=self.lr)

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

    def get_do_pre_load(self):
        '''
        Return True if the model should use a pre-load
        Return False otherwise
        :return:
        '''
        return self.do_preload