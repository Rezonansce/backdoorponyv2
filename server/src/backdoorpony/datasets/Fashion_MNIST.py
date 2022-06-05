import pickle
import os
import zipfile
import pandas as pd
import numpy as np


class Fashion_MNIST(object):

    def __init__(self, num_selection=500):
        self.num_selection = min(num_selection, 50000)

    def get_datasets(self):
        '''
        Returns the Stanford Dogs dataset
        :return: tuple containing training and test sets
        '''
        abs_path = os.path.abspath(__file__)
        file_directory = os.path.dirname(abs_path)
        zip_file = r'fashion_mnist/archive.zip'
        train_file = r'fashion-mnist_train.csv'
        test_file = r'fashion-mnist_test.csv'
        zf = zipfile.ZipFile(os.path.join(file_directory, zip_file))
        train_df = pd.read_csv(zf.open(train_file))
        test_df = pd.read_csv(zf.open(test_file))
        train_features = train_df[train_df.columns[1:]].to_numpy()
        test_features = test_df[test_df.columns[1:]].to_numpy() / 255.0
        indices = np.random.choice(50000, size=self.num_selection, replace=False)
        X_train = train_features[indices] / 255.0
        X_train = X_train.reshape(len(X_train), 1, 28, 28)
        test_features = test_features.reshape(len(test_features), 1, 28, 28)
        train_labels = train_df['label'].to_numpy()[indices]
        test_labels = test_df['label'].to_numpy()
        return (X_train, train_labels), (test_features, test_labels)





def unpickle(self, file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict