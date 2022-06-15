import os.path

import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.utils import preprocess
import torch
import os.path
from backdoorpony.classifiers.abstract_classifier import AbstractClassifier


class ImageClassifier(PyTorchClassifier, AbstractClassifier):
    def __init__(self, model, autoencoder=None):
        '''Initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on
        autoencoder:
            Autoencoder to attach to the model
        Returns
        ----------
        None
        '''
        # move the model to gpu if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        self.autoencoder = autoencoder
        super().__init__(
            model=model,
            clip_values=(0.0, 255.0),
            loss=model.get_criterion(),
            optimizer=model.get_opti(),
            input_shape=model.get_input_shape(),
            nb_classes=model.get_nb_classes()
        )

    def fit(self, x, y, use_pre_load=False, *args, **kwargs):
        '''Fits the classifier to the training data
        If the classifier was already trained, pre-load the state_dict
        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on
        use_pre_load:
            True if the model should be pre-loaded/ if there is no pre-load,
            the model will be fitted with the current data and saved as pre-load
            False if fit is used to poison/defend model

        Returns
        ----------
        None
        '''

        if use_pre_load:
            # Get relative paths to the pre-load directory
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
        # Else, fit the training set
        x_train = x
        y_train = y
        #print(np.shape(x_train))
        x_train, y_train = preprocess(x_train, y_train, nb_classes=super().model.get_nb_classes())
        x_train = np.float32(x_train)
        # TODO: Parameterize batch size and number of epochs
        super().fit(x_train, y_train, batch_size=16, nb_epochs=10)
        if use_pre_load:
            # Save the trained weights
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
        super().model.eval()
        if self.autoencoder is not None:
            x = self.autoencoder.predict(x)
        return super().predict(x.astype(np.float32))

    def set_autoencoder(self, autoencoder):
        '''
        Setter for the autoencoder
        :param autoencoder: The autoencoder defense attached to the classifier
        :return: None
        '''
        self.autoencoder = autoencoder

    def class_gradient(self, x, *args, **kwargs):
        return super().class_gradient(x)

    def get_model(self):
        '''
        Return the neural network model of the classifier
        :return: The neural network model
        '''
        return super().model
