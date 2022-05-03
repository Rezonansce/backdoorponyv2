import torch
import torch.nn as nn
import torch.optim as optim
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier

# optimizer = optim.SGD(model.parameters(), lr=1e-3)

# criterion = nn.BCEWithLogitsLoss()

def binary_accuracy(preds, y):
        '''
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        '''

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

class TextClassifier(AbstractClassifier, object):
    def __init__(self, model):
        '''Initiates the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=1e-3)

    def fit(self, train_data):
        '''Fit the classifier to the training data
        
        Parameters
        ----------
        train_data :
            Data that the classifier will be trained on

        Returns
        ----------
        None
        '''
        return self.train(train_data)

    def predict(self, inputs):
        '''Return the predicted classification of the input

        Parameters
        ----------
        inputs :
            The dataset the classifier should classify

        Returns
        ----------
        prediction : 
            Return format can be anything, as long as it is consistent between
            classifiers of the same category
        '''
        self.evaluate(inputs)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        return

    def train(self, iterator):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.model.train()
        
        for batch in iterator:
            
            self.optimizer.zero_grad()
                    
            predictions = self.model(batch.text).squeeze(1)
            
            loss = self.criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)
            
            loss.backward()
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, iterator):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                predictions = self.model(batch.text).squeeze(1)
                
                loss = self.criterion(predictions, batch.label)
                
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
