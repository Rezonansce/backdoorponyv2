'''
The below code generates a Convolutional Neural Network
for working with the Audio MNIST dataset.
'''
import torch.nn as nn
import torch.nn.functional as F

#__dataset__ = 'audio_vgd'
#__class_name__ = 'Audio_VGD_CNN'

__name__ = "Audio_VGD_CNN"
__category__ = 'audio'
__input_type__ = "audio"
__defaults__ = {
    'kernel_size': {
        'pretty_name': 'Kernel size',
        'default_value': [5],
        'info': 'Kernel size, should be < 28'
    },
    'hidden_layer_nodes': {
        'pretty_name': 'Hidden Layer Nodes',
        'default_value': [500],
        'info': 'This parameter can adjust the number of nodes in the last layer.'
    },
    'optim': {
        'pretty_name': 'Optimizer',
        'default_value': ['Adam'],
        'info': 'The optimizer used in the training process. Currently, only "Adam" and "SGD" are available.' +
            'If the input is not valid, Adam optimizer will be chosen.'
    },
    'num_train': {
        'pretty_name': 'Fraction of the training samples of a dataset',
        'default_value': [1],
        'info': 'Consists of 3000 samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset loaded for training'
    },
    'num_test': {
        'pretty_name': 'Fraction of the testing samples of a dataset',
        'default_value': [1],
        'info': 'Consists of 3000 samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset loaded for testing'
    }
}
__link__ = 'link to model page'
__info__ = '''A model that trains spectrogrammer input'''

class Audio_VGD_CNN(nn.Module):
    def __init__(self, model_parameters):
        '''Initiates a CNN geared towards the Audio MNIST dataset

        Returns
        ----------
        None
        '''
        super(Audio_VGD_CNN, self).__init__()

        self.kernel_size = model_parameters['kernel_size']['value'][0]
        self.hidden_layer_nodes = model_parameters['hidden_layer_nodes']['value'][0]
        self.optim = model_parameters['optim']['value'][0]
        if self.optim != "SGD":
            self.optim = "Adam"

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=self.kernel_size, stride=1)
        self.conv_2 = nn.Conv2d(
            in_channels=20, out_channels=60, kernel_size=self.kernel_size, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 60, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=self.hidden_layer_nodes)
        self.fc_3 = nn.Linear(in_features=self.hidden_layer_nodes, out_features=2)

        self.nb_class = 2

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 60)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return F.log_softmax(x, dim=1)