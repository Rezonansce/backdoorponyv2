import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__name__ = "AMAZON_CNN"
__input_type__ = "text"
__defaults__ = {
    'filter_sizes': {
        'pretty_name': 'list of filter sizes',
        'default_value': [1,2,3,4,5],
        'info': 'Different filters that will be applied in the convolutional layer of the network'
    },
    'drop_prob': {
        'pretty_name': 'dropout probability',
        'default_value': [0.5],
        'info': 'percentage of recurrent connections to CNN excluded from activation and weight updates'
    },
    'embedding_dim': {
        'pretty_name': 'Size of the embedding layer',
        'default_value': [300],
        'info': 'Dimension of the embedding layer'
    },
    'n_filters': {
        'pretty_name': 'number of filters per filter size',
        'default_value': [100],
        'info': 'A total number of filters applied for each filter size'
    },
    'num_train': {
        'pretty_name': 'Fraction of the training samples of a dataset',
        'default_value': [1],
        'info': 'Consists of 25000 training samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset'
    },
    'num_test': {
        'pretty_name': 'Fraction of the testing samples of a dataset',
        'default_value': [1],
        'info': 'Consists of 25000 test samples, choose between 0 and 1, where 0 corresponds to 0% and 1 corresponds to 100% of the dataset'
    }
}
__link__ = 'https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html'
__info__ = '''Convolutional Neural Network'''


class AMAZON_CNN(nn.Module):
    def __init__(self, vocab_size, model_parameters):
        '''Initiates a RNN geared towards the IMDB dataset

        Returns
        ----------
        None
        '''
        super().__init__()
        # dimension of the embedding layer
        embedding_dim = model_parameters['embedding_dim']['value'][0]
        # number of filters per filter size
        self.n_filters = model_parameters['n_filters']['value'][0]
        # list of filter sizes to be used in the convolutional layer
        self.filter_sizes = model_parameters['filter_sizes']['value']
        # output dimension
        self.output_dim = 1
        # dropout probability
        drop_prob = model_parameters['drop_prob']['value'][0]

        # an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = self.n_filters, kernel_size = (fs, embedding_dim)) for fs in self.filter_sizes])

        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(drop_prob)

        # linear layer for final binary classification evaluation
        self.linear = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)

        # sigmoid to activate the linear layer
        self.sigmoid = nn.Sigmoid()



    def init_hidden(self, batch_size, device):
        '''
        hidden layers are not required for the cnn
        '''


    # forward pass of the algorithm
    def forward(self, features, hid):
        '''
        Forward pass of the algorithm
        Parameters
        ----------
        features - inputs of the algorithm
        hid - (h0, c0) - a 2-tuple of hidden and cell states

        Returns
        -------
        a 2-tuple (sig_ret, hid)
        where sig_ret - labels
        and hid - updated hidden and cell states as a 2-tuple (h0, c0)
        '''
        batch_size = features.size(0)

        # pass through embedding layer
        embeds = self.embedding(features).unsqueeze(1) # shape: B x S x Feature

        # pass through convolutional layers
        conved = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]

        # pass through pooling layer to reduce dimensionality
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pass through dropout layer
        drop = self.dropout(torch.cat(pooled, dim = 1))

        # pass through linear layer
        ret = self.linear(drop)

        # pass through sigmoid
        sig_ret = self.sigmoid(ret)

        # reshape
        sig_ret = sig_ret.view(batch_size, -1)

        # get last labels batch
        sig_ret = sig_ret[:, -1]

        return sig_ret, hid