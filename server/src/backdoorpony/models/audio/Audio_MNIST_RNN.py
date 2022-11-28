'''
The below code generates a Convolutional Neural Network
for working with the Audio MNIST dataset.
'''
import torch.nn as nn
import torch.nn.functional as F

__name__ = "Audio_MNIST_RNN"
__category__ = 'audio'
__defaults_form__ = {}
__defaults_dropdown__ = {
    'optim': {
        'pretty_name': 'Optimizer',
        'default_value': ['Adam'],
        'possible_values': ['Adam', 'SGD'],
        'info': 'The optimizer used in the training process. Currently, only "Adam" and "SGD" are available.' +
            'If the input is not valid, Adam optimizer will be chosen.'
    }
}

__defaults_range__ = {
    'kernel_size': {
        'pretty_name': 'Kernel size',
        'default_value': [5],
        'minimum': 1,
        'maximum': 27,
        'info': 'Kernel size, should be < 28'
    },
    'hidden_layer_nodes': {
        'pretty_name': 'Hidden Layer Nodes',
        'default_value': [784],
        'minimum': 1,
        'maximum': 10000,
        'info': 'This parameter can adjust the number of nodes in the last layer.'
    },
    'num_train': {
        'pretty_name': 'Fraction of the training samples of a dataset',
        'default_value': [1.0],
        'minimum': 0.0,
        'maximum': 1.0,
        'info': 'Consists of 3000 samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset loaded for training'
    },
    'num_test': {
        'pretty_name': 'Fraction of the testing samples of a dataset',
        'default_value': [1.0],
        'minimum': 0.0,
        'maximum': 1.0,
        'info': 'Consists of 3000 samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset loaded for testing'
    }
}
__input_type__ = "audio"

__link__ = 'link to model page'
__info__ = '''A model that trains spectrogrammer input'''
# __dataset__ = 'audio_mnist'
# __class_name__ = 'Audio_MNIST_RNN'


class Audio_MNIST_RNN(nn.Module):
    def __init__(self, model_parameters):
        '''Initiates a CNN geared towards the Audio MNIST dataset

        Returns
        ----------
        None
        '''
        super(Audio_MNIST_RNN, self).__init__()
        self.kernel_size = model_parameters['kernel_size']['value'][0]
        self.hidden_layer_nodes = model_parameters['hidden_layer_nodes']['value'][0]
        self.optim = model_parameters['optim']['value'][0]
        if self.optim != "SGD":
            self.optim = "Adam"

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=self.kernel_size, stride=1)
        self.conv_2 = nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=self.kernel_size, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 50, out_features=self.hidden_layer_nodes)
        self.fc_2 = nn.Linear(in_features=self.hidden_layer_nodes, out_features=10)



        self.nb_class = 10

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)
