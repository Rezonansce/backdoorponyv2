from copy import deepcopy

import torch
from backdoorpony.classifiers.GraphClassifier import GraphClassifier
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.classifiers.TextClassifier import TextClassifier
from backdoorpony.datasets.IMDB import IMDB
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.MUTAG import MUTAG
from backdoorpony.models.image.MNIST_CNN import MNIST_CNN
from backdoorpony.models.text.IMDB_RNN import IMDB_RNN
from backdoorpony.models.graph.zaixizhang import graphcnn


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
                }
            },
            'text': {
                'classifier': TextClassifier,
                'IMDB': {
                    'dataset': IMDB,
                    'model': IMDB_RNN,
                    'link': 'https://ai.stanford.edu/~amaas/data/sentiment/',
                    'info': 'The IMDB dataset consists of 50,000 movie reviews from IMDB users. These reviews are in text format and are labelled as either positive (class 1) or negative (class 0). Each review is encoded as a sequence of integer indices, each index corresponding to a word. The value of each index is represented by its frequency within the dataset. For example, integer “3” encodes the third most frequent word in the data. The training and the test sets contain 25,000 reviews, respectively.'

                }
            },
            'audio': {
                'classifier': ...
            },
            'graph': {
                'classifier': GraphClassifier,
                'MUTAG': {
                    'dataset': MUTAG,
                    'model': graphcnn,
                    'link': None,
                    'info': None
                }
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
        model = self.options[type][dataset]['model']()
        
        if file_model != None:
            name = file_model.filename.split('.', 1) #remove filename extension
            file_model.save('data/pth/' + name[0] + '.model.pth')
            state_dict = torch.load('data/pth/' + name[0] + '.model.pth')
            model.load_state_dict(state_dict)

        self.train_data, self.test_data = self.options[type][dataset]['dataset']().get_datasets()
        self.classifier = self.options[type]['classifier'](model)
        x, y = self.train_data
        self.classifier.fit(x, y)
    
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
