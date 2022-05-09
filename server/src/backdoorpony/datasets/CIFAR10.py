import pickle
import os
import numpy as np
# TODO: ZIP THE CIFAR10 Directory
class CIFAR10(object):


    def __init__(self, num_selection = 7500):
        self.num_selection = num_selection

    def get_datasets(self):
        abs_path = os.path.abspath(__file__)
        file_directory = os.path.dirname(abs_path)
        batch_1_file = r'cifar10/batch1'
        batch_2_file = r'cifar10/data_batch_2'
        batch_3_file = r'cifar10/data_batch_3'
        batch_4_file = r'cifar10/data_batch_4'
        batch_5_file = r'cifar10/data_batch_5'
        meta_file = r'cifar10/batches.meta'
        test_file = r'cifar10/test_batch'
        # meta_data['label_names'] has the names of each of the ten labels
        meta_data = self.unpickle(os.path.join(file_directory, meta_file))
        batch_1 = self.unpickle(os.path.join(file_directory, batch_1_file))
        batch_2 = self.unpickle(os.path.join(file_directory, batch_2_file))
        batch_3 = self.unpickle(os.path.join(file_directory, batch_3_file))
        batch_4 = self.unpickle(os.path.join(file_directory, batch_4_file))
        batch_5 = self.unpickle(os.path.join(file_directory, batch_5_file))
        X_train = np.concatenate((batch_1[b'data'], batch_2[b'data']
                                  , batch_3[b'data'], batch_4[b'data']
                                  , batch_5[b'data']), axis=0)
        indices = np.random.choice(50000, size=self.num_selection)
        X_train = X_train[indices]
        X_train = X_train / 255.0
        Y_train = np.concatenate((batch_1[b'labels'], batch_2[b'labels']
                                  , batch_3[b'labels'], batch_4[b'labels']
                                  , batch_5[b'labels']), axis=0)
        Y_train = Y_train[indices]
        test_batch = self.unpickle(os.path.join(file_directory, test_file))
        y_test = test_batch[b'labels']
        x_test = test_batch[b'data'] / 255.0
        # Reshape the whole image data
        X_train = X_train.reshape(len(X_train), 3, 32, 32).astype(np.float32)
        # X_train = X_train.transpose(0, 2, 3, 1)
        x_test = x_test.reshape(len(x_test), 3, 32, 32).astype(np.float32)
        # x_test = x_test.transpose(0, 2, 3, 1)
        train_data = (X_train, Y_train)
        test_data = (x_test, y_test)
        return train_data, test_data

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

