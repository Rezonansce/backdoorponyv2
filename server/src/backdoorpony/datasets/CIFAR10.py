import pickle
import os
import numpy as np
import zipfile
import pandas as pd

# TODO: ZIP THE CIFAR10 Directory
class CIFAR10(object):


    def __init__(self, num_selection = 25000):
        self.num_selection = min(num_selection, 50000)

    def get_datasets(self):
        '''
        Loads the CIFAR-10 dataset

        :return: tuple containing training and test sets
        '''
        abs_path = os.path.abspath(__file__)
        file_directory = os.path.dirname(abs_path)
        zip_file = r'cifar10/cifar10.zip'
        batch_1_file = r'data_batch_1'
        batch_2_file = r'data_batch_2'
        batch_3_file = r'data_batch_3'
        batch_4_file = r'data_batch_4'
        batch_5_file = r'data_batch_5'
        test_file = r'test_batch'
        # meta_data['label_names'] has the names of each of the ten labels
        with zipfile.ZipFile(os.path.join(file_directory, zip_file), 'r') as zip_ref:
            # Extract zipped files
            batch_1 = pd.read_pickle(zip_ref.open(batch_1_file))
            batch_2 = pd.read_pickle(zip_ref.open(batch_2_file))
            batch_3 = pd.read_pickle(zip_ref.open(batch_3_file))
            batch_4 = pd.read_pickle(zip_ref.open(batch_4_file))
            batch_5 = pd.read_pickle(zip_ref.open(batch_5_file))
            test_batch = pd.read_pickle(zip_ref.open(test_file))
            zip_ref.close()
        # Concatenate all the batches
        X_train = np.concatenate((batch_1['data'], batch_2['data']
                                  , batch_3['data'], batch_4['data']
                                  , batch_5['data']), axis=0)
        # Select only the number of samples mentioned by the constructor for the training data
        indices = np.random.choice(50000, size=self.num_selection, replace=False)
        X_train = X_train[indices]
        X_train = X_train / 255.0
        Y_train = np.concatenate((batch_1['labels'], batch_2['labels']
                                  , batch_3['labels'], batch_4['labels']
                                  , batch_5['labels']), axis=0)
        Y_train = Y_train[indices]
        y_test = test_batch['labels']
        x_test = test_batch['data'] / 255.0
        # Reshape the whole image data
        # 3 channels for RGB, 32 pixels width, 32 pixels height
        X_train = X_train.reshape(len(X_train), 3, 32, 32)
        x_test = x_test.reshape(len(x_test), 3, 32, 32)
        train_data = (X_train, Y_train)
        test_data = (x_test, y_test)
        return train_data, test_data

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

