import numpy as np
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.utils import preprocess
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier


class AudioClassifier(PyTorchClassifier, AbstractClassifier):
    def __init__(self, model):
        '''Initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        criterion = nn.CrossEntropyLoss()
        opti = optim.Adam(model.parameters(), lr=0.01)
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=criterion,
            optimizer=opti,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

    def fit(self, x, y, *args, **kwargs):
        '''Fits the classifier to the training data
        First normalises the data and transform it to the format used by PyTorch.

        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on

        Returns
        ----------
        None
        '''
        x_train = x
        y_train = y
        x_train, y_train = preprocess(x_train, y_train)
        x_train = np.expand_dims(x_train, axis=3)
        x_train = np.transpose(x_train, (0, 3, 2, 1)).astype(np.float32)
        super().fit(x_train, y_train, 64, 3)

    def predict(self, x, *args, **kwargs):
        '''Classifies the given input

        Parameters
        ----------
        x :
            The dataset the classifier should classify

        Returns
        ----------
        prediction :
            Return format is a numpy array with the probability for each class
        '''
        x = np.expand_dims(x, axis=3)
        x = np.transpose(x, (0, 3, 2, 1)).astype(np.float32)
        return super().predict(x)

    def class_gradient(self, x, *args, **kwargs):
        x = np.expand_dims(x, axis=3)
        x = np.transpose(x, (0, 3, 2, 1)).astype(np.float32)
        return super().class_gradient(x)
