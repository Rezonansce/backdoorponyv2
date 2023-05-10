import torch
import torch.nn as nn

__name__ = "IMDB_LSTM_RNN"
__input_type__ = "text"
__defaults__ = {
    'bidirectional': {
        'pretty_name': 'Make LSTM layers bidirectional (True/False)',
        'default_value': ["True"],
        'info': 'True - LSTM passes data both to the future and to the past, False - only to the future'
    },
    'drop_prob': {
        'pretty_name': 'dropout probability',
        'default_value': [0.5],
        'info': 'percentage of recurrent connections to LSTM excluded from activation and weight updates'
    },
    'embedding_dim': {
        'pretty_name': 'Size of the embedding layer',
        'default_value': [300],
        'info': 'Dimension of the embedding layer'
    },
    'hidden_dim': {
        'pretty_name': 'Number of hidden layers of lstm',
        'default_value': [128],
        'info': 'the total number of hidden nodes in lstm layers'
    },
    'lstm_layers': {
        'pretty_name': 'Number of (stacked) LSTM layers',
        'default_value': [2],
        'info': 'the total number of (stacked) lstm-layers (even if bidirectional)'
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
    },
    'pre_load': {
        'pretty_name': 'Preload Model',
        'default_value': [False],
        'info': 'True if you would like to use a pre-trained model with selected hyperparameters(if exists, created otherwise). False otherwise.'
    },
}
__link__ = 'https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html'
__info__ = '''LSTM with a head sigmoid layer'''


class IMDB_LSTM_RNN(nn.Module):
    def __init__(self, vocab_size, model_parameters):
        '''Initiates a RNN geared towards the IMDB dataset

        Returns
        ----------
        None
        '''
        super().__init__()
        # dimension of the embedding layer
        embedding_dim = model_parameters['embedding_dim']['value'][0]
        # the total number of stacked lstm-layers
        self.lstm_layers = lstm_layers = model_parameters['lstm_layers']['value'][0]
        # number of hidden layers of lstm
        self.hidden_dim = hidden_dim_linear = model_parameters['hidden_dim']['value'][0]
        # output dimension
        self.output_dim = 1
        # if set to true, becomes bidirectional
        bidirectional = model_parameters['bidirectional']['value'][0].lower() == "true"
        # lstm dropout probability
        drop_prob = model_parameters['drop_prob']['value'][0]

        # an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # create a (stacked if lstm_layers > 1) lstm model
        if bidirectional:
            lstm_layers = round(lstm_layers/2)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)

        # if lstm is bidirectional, there needs to be twice as much hidden nodes
        if bidirectional:
            hidden_dim_linear *= 2

        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(drop_prob)

        # linear layer for final binary classification evaluation
        self.linear = nn.Linear(hidden_dim_linear, self.output_dim)

        # sigmoid to activate the linear layer
        self.sigmoid = nn.Sigmoid()

        self.do_preload = model_parameters['pre_load']['value'][0]

        # use all parameters relevant to the model to create a unique path
        self.path = 'imdb-lstm-rnn' + \
                    '_' + str(bidirectional) + \
                    '_' + str(drop_prob) + \
                    '_' + str(embedding_dim) + \
                    '_' + str(hidden_dim_linear) + \
                    '_' + str(lstm_layers) + \
                    '_' + str(model_parameters['num_train']['value'][0]) + \
                    '_' + str(model_parameters['num_test']['value'][0]) + \
                    '.pt'

    def get_do_pre_load(self):
        return self.do_preload

    def get_path(self):
        return self.path

    def init_hidden(self, batch_size, device):
        '''
        Initializes hidden and cell states
        Parameters
        ----------
        batch_size - number of entries in one batch
        device - cpu or gpu to use for computations

        Returns
        -------
        hid - a 2-tuple (h0, c0) of hidden and cell states
        '''
        # initialize hidden state
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)

        # initialize cell state
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)
        hid = (h0, c0)
        return hid

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
        embeds = self.embedding(features) # shape: B x S x Feature

        # pass through lstm layer
        lstm_ret, hid = self.lstm(embeds, hid)

        # pass through dropout layer
        ret = self.dropout(lstm_ret)

        # pass through linear layer
        ret = self.linear(ret)

        # pass through sigmoid
        sig_ret = self.sigmoid(ret)

        # reshape
        sig_ret = sig_ret.view(batch_size, -1)

        # get last labels batch
        sig_ret = sig_ret[:, -1]

        return sig_ret, hid