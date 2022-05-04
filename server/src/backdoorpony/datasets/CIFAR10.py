import pickle

class CIFAR10(object):


    def __init__(self, num_selection = 7500):
        self.num_selection = num_selection

    def get_datasets(self):
        batch_1_file = r'C:\Users\vladm\PycharmProjects\backdoor-pony-v2\server\src\backdoorpony\data\cifar10\data_batch_1'
        batch_1 = self.unpickle(batch_1_file)
        meta_file = r'C:\Users\vladm\PycharmProjects\backdoor-pony-v2\server\src\backdoorpony\data\cifar10\batches.meta'
        # meta_data['label_names'] has the names of each of the ten labels
        meta_data = self.unpickle(meta_file)
        X_train = batch_1['data']
        # Reshape the whole image data
        X_train = X_train.reshape(len(X_train), 3, 32, 32)
        X_train = X_train.transpose(0, 2, 3, 1)
        Y_train = batch_1['labels']
        test_file = r'C:\Users\vladm\PycharmProjects\backdoor-pony-v2\server\src\backdoorpony\data\cifar10\test_batch'
        test_batch = self.unpickle(test_file)
        x_test = test_batch['data']
        x_test = x_test.reshape(len(x_test), 3, 32, 32)
        x_test = x_test.transpose(0, 2, 3, 1)
        y_test = test_batch['labels']
        train_data = (X_train, Y_train)
        test_data = (x_test, y_test)
        return train_data, test_data


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

