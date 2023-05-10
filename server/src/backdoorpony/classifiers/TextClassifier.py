import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def binary_accuracy(preds, y):
        '''
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        '''

        # round predictions to the closest integer
        rounded_preds = torch.round(preds.squeeze())

        # convert into float for later division
        correct = (rounded_preds == y).float()

        # calculate accuracy
        acc = correct.sum() / len(correct)
        return acc

class TextClassifier(AbstractClassifier, object):
    def __init__(self, model, vocab, learning_rate):
        '''Initiates the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        # self.optimizer = optim.SGD(model.parameters(), lr=1e-3)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # vocabulary
        self.vocab = vocab


    def fit(self, x, y, poison=False, *args, **kwargs):
        '''Fit the classifier to the training data
        
        Parameters
        ----------
        train_data :
            x and y - fatures and labels that the classifier will be trained on

        Returns
        ----------
        evaluation metrics - (loss, accuracy) as a 2-tuple
        '''

        # Check if the user asked for a pre-loaded model
        if self.model.get_do_pre_load() and not poison:
            # Get relative paths to the pre-load directory
            abs_path = os.path.abspath(__file__)
            file_directory = os.path.dirname(abs_path)
            parent_directory = os.path.dirname(file_directory)
            target_path = r'models/text/pre-load'
            final_path = os.path.join(parent_directory, target_path
                                      , self.model.get_path())
            # If there is a pretrained model, just load it
            if os.path.exists(final_path):
                self.model.load_state_dict(torch.load(final_path))
                return

        batch_size = 250

        # prepare the data for iterating batches
        train_tensor = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size, drop_last=True)

        # train the classifier and get evaluation metrics
        evmetrics = self.train(train_loader, batch_size, numEpochs=5)

        print("Train: ", evmetrics)

        # empty reference to the memory occupied by variables previously
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

        if self.model.get_do_pre_load() and not poison:
            # Save the trained weights
            torch.save(self.model.state_dict(), final_path)

        return evmetrics

    def predict(self, x, *args, **kwargs):
        '''Return the predicted classification of the input

        Parameters
        ----------
        inputs :
            x - the dataset the classifier should classify

        Returns
        ----------
        prediction : 
            Return 1d numpy array of predictions
        '''
        # batch size dynamically chosen for prediction, but has to be between 1 and 500
        # in order to avoid cuda memory outage when using a gpu
        batch_size = min(max(round(len(x)/10), 1), 250)

        # Initialize data loaders to iterate through the batches
        pred_tensor = TensorDataset(torch.from_numpy(x))
        pred_loader = DataLoader(pred_tensor, shuffle=False, batch_size=batch_size, drop_last=True)

        outs = []

        # initialize hidden and cell states
        h = self.model.init_hidden(batch_size, self.device)

        # shaping
        if h is not None:
            h = tuple([x.data for x in h])

        # run the prediction process on the whole dataset, ignore hidden state changes
        with torch.no_grad():
            for idx, features in tqdm(enumerate(pred_loader)):
                (features,) = features
                features = features.to(self.device)
                output, _ = self.model(features, h)

                for value in output.detach().cpu().numpy():
                    outs.append(round(value))

        # empty reference to the memory occupied by variables previously
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

        # return a numpy array of predictions
        return numpy.array(outs)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        return

    def set_device(self, device):
        self.device = device
        return

    def get_device(self, device):
        return self.device

    def train_one_epoch(self, train_loader, h):
        '''

        Parameters
        ----------
        train_loader - dataloader to iterate through the batches
        h - hidden state

        Returns
        -------
        (epoch_locc, epoch_acc, h) as a 3-tuple
        '''
        # initialize loss and accuracy for current epoch
        epoch_loss = 0
        epoch_acc = 0
        
        self.model.train()

        for features, labels in tqdm(train_loader):
            # create new variables to prevent backpropagating through the whole history
            if h is not None:
                h = tuple([x.data for x in h])

            # pytorch accumulates gradients, so reset to zero
            self.model.zero_grad()

            # move to correct device
            features, labels = features.to(self.device), labels.to(self.device)

            # predict
            predictions, h = self.model(features, h)

            # calculate loss
            loss = self.criterion(predictions.squeeze(), labels.float())

            # backpropagate
            loss.backward()

            # calculate accuracy
            acc = binary_accuracy(predictions.squeeze(), labels.float())
            
            self.optimizer.step()

            # update metrics
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader), h

    def train(self, train_loader, batch_size, numEpochs):
        '''

        Parameters
        ----------
        train_loader - dataloader to iterate through the batches
        batch_size - number of entries in one batch
        numEpochs - number of epochs to train on

        Returns
        -------
        (epoch_loss, epoch_acc) as a 2-tuple
        '''
        # initialize hidden and cell states
        h = self.model.init_hidden(batch_size, self.device)

        # initialize epoch loss and accuracy
        epoch_loss = epoch_acc = 0

        # run training numEpochs number of times
        for i in range(numEpochs):
            print("Epoch ", i+1, ":")

            # run one epoch training
            epoch_loss, epoch_acc, h = self.train_one_epoch(train_loader, h)

            print("Epoch acc: ", epoch_acc)
        return epoch_loss, epoch_acc

    def class_gradient(self, x, *args, **kwargs):
        return super().class_gradient(x)