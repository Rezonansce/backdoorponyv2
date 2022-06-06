'''
Load the MNIST dataset (raw images) for use in attacks and or defences.

:param num_selection: The size of the training set. Default 7500
:return (x_raw_train, y_raw_test), (x_raw_test, y_raw_test), min, max
'''
import numpy as np
from art.utils import load_mnist

class MNIST(object):
    def __init__(self, num_selection=3000):
        '''Should initiate the dataset

        Returns
        ----------
        None
        '''
        self.num_selection = min(50000, num_selection)

    def get_datasets(self):
        '''Should return the training data and testing data

        Returns
        ----------
        train_data :
            A tuple with two numpy arrays: one with the sample and one with the corresponding labels
        test_data :
            A tuple with two numpy arrays: one with the sample and one with the corresponding labels
        '''
        (x_raw_train, y_raw_train), (x_raw_test, y_raw_test), _, _ = self.get_data()
        # Expand the dimensions of the input, from 1 to 3
        x_train = np.expand_dims(x_raw_train, axis=3)
        # Transpose the input
        x_train = np.transpose(x_train, (0, 3, 2, 1)).astype(np.float32)
        x_test = np.expand_dims(x_raw_test, axis=3)
        x_test = np.transpose(x_test, (0, 3, 2, 1)).astype(np.float32)
        train_data = (x_train, y_raw_train)
        test_data = (x_test, y_raw_test)
        return train_data, test_data

    def get_data(self):
        '''
        Get the raw MNIST dataset.
        Automatically creates a split between train and test data.

        Returns:
            x_raw_train: The raw training data.
            y_raw_train: The labels for the raw training data.
            x_raw_test: The raw test data.
            y_raw_test: The labels for the raw test data.
        '''
        # Download and split the MNIST dataset into train and test.
        (x_raw, y_raw), (x_raw_test, y_raw_test), min, max = load_mnist(raw=True)

        # Random Selection (only take a part of the complete MNIST training set):
        n_train = np.shape(x_raw)[0]
        random_selection_indices = np.random.choice(
            n_train, self.num_selection)
        x_raw_train = x_raw[random_selection_indices]
        y_raw_train = y_raw[random_selection_indices]

        return (x_raw_train, y_raw_train), (x_raw_test, y_raw_test), min, max
