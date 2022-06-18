import torch.nn as nn
import torch.optim as optim

class AutoencoderCNN(nn.Module):

    def __init__(self, input_shape, path):
        super(AutoencoderCNN, self).__init__()
        self.crit = nn.MSELoss()
        self.path = path + "_autoencoder"
        self.nb_classes = input_shape[0] * input_shape[1] * input_shape[2]
        self.input_shape = input_shape
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_shape[0] * input_shape[1] * input_shape[2]),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_opti(self):
        '''
        Get the optimizer used for MNIST.

        :return: Adam optimizer, with learning rate = 0.01
        '''
        return optim.Adam(self.parameters(), lr=1e-1, weight_decay=1e-8)

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
