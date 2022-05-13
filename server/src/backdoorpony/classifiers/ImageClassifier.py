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
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model = model.to(device)
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=model.get_criterion(),
            optimizer=model.get_opti(),
            input_shape=model.get_input_shape(),
            nb_classes=model.get_nb_classes()
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
        abs_path = os.path.abspath(__file__)
        file_directory = os.path.dirname(abs_path)
        parent_directory = os.path.dirname(file_directory)
        target_path = r'models/image/pre-load'
        final_path = os.path.join(parent_directory, target_path
                                  , super().model.get_path())
        if os.path.exists(final_path):
            super().model.load_state_dict(torch.load(final_path))
            return
        x_train = x
        y_train = y
        x_train, y_train = preprocess(x_train, y_train)
        # x_train = np.expand_dims(x_train, axis=3)
        # x_train = np.transpose(x_train, (0, 3, 2, 1)).astype(np.float32)
        super().fit(x_train, y_train, batch_size=4, nb_epochs=5)
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
        # x = np.expand_dims(x, axis=3)
        # x = np.transpose(x, (0, 3, 2, 1)).astype(np.float32)
        print(x)
        print(np.shape(x))
        return super().predict(x)

    def class_gradient(self, x, *args, **kwargs):
        # x = np.expand_dims(x, axis=3)
        # x = np.transpose(x, (0, 3, 2, 1)).astype(np.float32)
        return super().class_gradient(x)

