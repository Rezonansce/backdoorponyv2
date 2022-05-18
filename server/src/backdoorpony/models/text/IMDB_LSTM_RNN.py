import torch
import torch.nn as nn


class IMDB_LSTM_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_layers, hidden_dim, output_dim, bidirectional = False, drop_prob = 0.5):
        '''Initiates a RNN geared towards the IMDB dataset
        
        Returns
        ----------
        None
        '''
        super().__init__()
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # create a (stacked if lstm_layers > 1) lstm model
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=round(lstm_layers/2), batch_first=True, bidirectional=bidirectional)

        # if lstm is bidirectional, there needs to be twice as much hidden nodes
        if bidirectional:
            hidden_dim *= 2

        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(drop_prob)

        # linear layer for final binary classification evaluation
        self.linear = nn.Linear(hidden_dim, output_dim)

        # sigmoid to activate the linear layer
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size, device):
        # initialize hidden state
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)

        # initialize cell state
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(device)
        hid = (h0, c0)
        return hid

    # forward pass of the algorithm
    def forward(self, features, hid):
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