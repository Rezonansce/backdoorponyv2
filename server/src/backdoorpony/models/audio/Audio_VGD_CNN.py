'''
The below code generates a Convolutional Neural Network
for working with the Audio MNIST dataset.
'''
import torch.nn as nn
import torch.nn.functional as F

__dataset__ = 'audio_vgd'
__class_name__ = 'Audio_VGD_CNN'


class Audio_VGD_CNN(nn.Module):
    def __init__(self):
        '''Initiates a CNN geared towards the Audio MNIST dataset

        Returns
        ----------
        None
        '''
        super(Audio_VGD_CNN, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(
            in_channels=20, out_channels=60, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 60, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=100)
        self.fc_3 = nn.Linear(in_features=100, out_features=2)

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
