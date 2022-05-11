import os.path

import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.utils import preprocess
import torch
from numba.cuda import jit

from backdoorpony.classifiers.abstract_classifier import AbstractClassifier


class ImageClassifier(PyTorchClassifier, AbstractClassifier):
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=model.get_criterion(),
            optimizer=model.get_opti(),
            input_shape=model.get_input_shape(),
            nb_classes=model.get_nb_classes()
        )

    def fit(self, x, y, first_training=False, *args, **kwargs):
        '''Fits the classifier to the training data
        If the classifier was already trained, pre-load the state_dict
        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on
        first_training:
            True if model is fitted with the initial data
            False if fit is used to poison/defend model

        Returns
        ----------
        None
        '''
        # Get relative paths to the pre-load directory
        if first_training:
            abs_path = os.path.abspath(__file__)
            file_directory = os.path.dirname(abs_path)
            parent_directory = os.path.dirname(file_directory)
            target_path = r'models/image/pre-load'
            final_path = os.path.join(parent_directory, target_path
                                      , super().model.get_path())
            # If there is a pretrained model, just load it
            if os.path.exists(final_path):
                super().model.load_state_dict(torch.load(final_path))
                return
        # Else, fit the training set and save it
        x_train = x
        y_train = y
        print(np.shape(x_train))
        x_train, y_train = preprocess(x_train, y_train, super().model.get_nb_classes())
        x_train = np.float32(x_train)
        # TODO: Broadcast batch_size and nb_epochs
        super().fit(x_train, y_train, batch_size=1, nb_epochs=5)
        if first_training:
            torch.save(super().model.state_dict(), final_path)


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
        return super().predict(x.astype(np.float32))

    def class_gradient(self, x, *args, **kwargs):
        return super().class_gradient(x)

