from copy import deepcopy
import torch
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.classifiers.TextClassifier import  TextClassifier
from backdoorpony.classifiers.AudioClassifier import AudioClassifier

from backdoorpony.datasets.Fashion_MNIST import Fashion_MNIST
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.audio_MNIST import Audio_MNIST
from backdoorpony.datasets.CIFAR10 import CIFAR10
from backdoorpony.datasets.IMDB import IMDB

from backdoorpony.models.image.Fashion_MNIST.FMNIST_CNN import FMNIST_CNN
from backdoorpony.models.audio.Audio_MNIST_RNN import Audio_MNIST_RNN
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.text.IMDB_LSTM_RNN import IMDB_LSTM_RNN
from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN




class Loader():

    def __init__(self, debug=False):
        '''Initiates a loader
        The loader is capable of loading/creating the classifier

        Parameters
        ----------
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        None

        '''
        self.classifier = None
        self.train_data = None
        self.test_data = None
        self.options = {
            'image': {
                'classifier': ImageClassifier,
                'MNIST': {
                    'dataset': MNIST,
                    'model': MNIST_CNN,
                    'link': 'http://yann.lecun.com/exdb/mnist/',
                    'info': 'The MNIST, or Modified National Institute of Standards and Technology, database comprises datasets of handwritten digit images. It is vastly used in machine learning for training and testing. The training set contains 60,000 examples, and the test set contains 10,000 examples.'
                },
                'CIFAR10': {
                    'dataset': CIFAR10,
                    'model': CifarCNN,
                    'link': 'https://www.cs.toronto.edu/~kriz/cifar.html',
                    'info': 'The CIFAR10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.'
                },
                'Fashion_MNIST': {
                    'dataset': Fashion_MNIST,
                    'model': FMNIST_CNN,
                    'link': 'https://github.com/zalandoresearch/fashion-mnist',
                    'info': 'Fashion-MNIST is a dataset of Zalando\'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. '
                }
            },
            'audio': {
                'classifier': AudioClassifier,
                'Audio_MNIST': {
                    'dataset': Audio_MNIST,
                    'model': Audio_MNIST_RNN,
                    'link': None,
                    'info': 'TODO ADD'
                }
            },
            'text': {
                'classifier': TextClassifier,
                'IMDB': {
                    'dataset': IMDB,
                    'model': IMDB_LSTM_RNN,
                    'link': 'https://ai.stanford.edu/~amaas/data/sentiment/',
                    'info': 'The IMDB dataset consists of 50,000 movie reviews from IMDB users. These reviews are in text format and are labelled as either positive (class 1) or negative (class 0). Each review is encoded as a sequence of integer indices, each index corresponding to a word. The value of each index is represented by its frequency within the dataset. For example, integer “3” encodes the third most frequent word in the data. The training and the test sets contain 25,000 reviews, respectively.'

                }
            },
            'graph': {
                'classifier': ...
            }
        }
        return None

    def get_datasets(self):
        '''Returns the datasets that are currently implemented

        Returns
        ----------
        Dictionary with the datasets categorised by the available types
        Takes the following shape:
        {
            'image': {
                'MNIST': {
                    'pretty_name': MNIST,
                    'info': 'Info on MNIST'
                },
                'CIFAR10': {
                    'pretty_name': CIFAR10,
                    'info': 'Info on CIFAR10'
                }
            },
            'text': {
                'IMDB': {
                    'pretty_name': IMDB,
                    'info': 'Info on IMDB'
                }
            },
            'audio': {},
            'graph': {}
        }
        '''
        sets = {}
        for type, dataset in self.options.items():
            contents = {}
            print(dataset)
            for name, attributes in dataset.items():
                if(name != 'classifier'):
                    contents.update({name: {'pretty_name': name, 'link': attributes['link'], 'info': attributes['info']}})
            sets[type] = contents

        return sets

    def make_classifier(self, type, dataset, file_model=None, debug=False):
        '''Creates the classifier corresponding to the input

        Parameters
        ----------
        type :
            Input type the classifier will act on, as defined by self.options
        dataset :
            The dataset the classifier will be fit to
        file_model :
            File (in .pth form) of the model the classifier will be based on
            Optional, if not set (or set to None) will use built-in classifier, as
            defined by self.options
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        None
        '''

        if type == "text":
            self.train_data, self.test_data, vocab = self.options[type][dataset]['dataset']().get_datasets()
            vocab_size = len(vocab) + 1
            print("Vocab size: ", vocab_size)
            embedding_dim = 10
            lstm_layers = 2
            hidden_dim = 16
            output_dim = 1
            model = self.options[type][dataset]['model'](vocab_size, embedding_dim, lstm_layers, hidden_dim, output_dim, True)

            self.classifier = self.options[type]['classifier'](model)
            x, y = self.train_data
            self.classifier.fit(x, y)
            return

        model = self.options[type][dataset]['model']()

        if file_model is not None:
            name = file_model.filename.split('.', 1) #remove filename extension
            file_model.save('data/pth/' + name[0] + '.model.pth')
            state_dict = torch.load('data/pth/' + name[0] + '.model.pth')
            model.load_state_dict(state_dict)

        self.train_data, self.test_data = self.options[type][dataset]['dataset']().get_datasets()
        self.classifier = self.options[type]['classifier'](model)
        x, y = self.train_data

        self.classifier.fit(x, y, use_pre_load=True)


    def get_classifier(self, debug=False):
        '''Gets the classifier if one has been made

        Returns
        ----------
        Returns the trained classifier if one has been made, else returns None
        '''
        return self.classifier

    def get_copy_classifier(self, debug=False):
        '''Gets a copy of the classifier if one has been made

        Returns
        ----------
        Returns a copy of the trained classifier if one has been made, else returns None
        '''
        return deepcopy(self.classifier)

    def get_train_data(self, debug=False):
        '''Gets the training data

        Returns
        ----------
        Returns the training data if it has been instantiated, else returns None
        '''
        return self.train_data

    def get_test_data(self, debug=False):
        '''Gets the validation data

        Returns
        ----------
        Returns the validation data if it has been instantiated, else returns None
        '''
        return self.test_data
