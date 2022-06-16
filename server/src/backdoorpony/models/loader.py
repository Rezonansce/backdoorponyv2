from copy import deepcopy
import torch
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.classifiers.TextClassifier import  TextClassifier
from backdoorpony.classifiers.AudioClassifier import AudioClassifier
from backdoorpony.classifiers.GraphClassifierNew import GraphClassifier

from backdoorpony.datasets.Fashion_MNIST import Fashion_MNIST
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.audio_MNIST import Audio_MNIST
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.graph.zaixizhang.gcnn_MUTAG import Gcnn_MUTAG
from backdoorpony.datasets.CIFAR10 import CIFAR10
from backdoorpony.datasets.IMDB import IMDB
from backdoorpony.datasets.MUTAG import MUTAG
from backdoorpony.datasets.AIDS import AIDS
from backdoorpony.datasets.Mutagenicity import Mutagenicity
from backdoorpony.datasets.Yeast import Yeast

from backdoorpony.models.image.Fashion_MNIST.FMNIST_CNN import FMNIST_CNN
from backdoorpony.models.audio.Audio_MNIST_RNN import Audio_MNIST_RNN
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.text.IMDB_LSTM_RNN import IMDB_LSTM_RNN
from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN
from backdoorpony.models.graph.gta.AIDS.AIDS_gcn import AIDS_gcn
from backdoorpony.models.graph.gta.Mutagenicity.Mutagenicity_gcn import Mutagenicity_gcn
from backdoorpony.models.graph.gta.Yeast.Yeast_gcn import Yeast_gcn




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
                'classifier': GraphClassifier,
                'AIDS': {
                    'dataset': AIDS,
                    'model': AIDS_gcn,
                    'link': "https://paperswithcode.com/dataset/aids",
                    'info': "AIDS is a graph dataset. It consists of 2000 graphs representing molecular compounds which are constructed from the AIDS Antiviral Screen Database of Active Compounds. It contains 4395 chemical compounds, of which 423 belong to class CA, 1081 to CM, and the remaining compounds to CI."
                },
                'Mutagenicity': {
                    'dataset': Mutagenicity,
                    'model': Mutagenicity_gcn,
                    'link': "https://paperswithcode.com/dataset/mutagenicity",
                    'info': "Mutagenicity is a chemical compound dataset of drugs, which can be categorized into two classes: mutagen and non-mutagen."
                },
                'Yeast': {
                    'dataset': Yeast,
                    'model': Yeast_gcn,
                    'link': "https://paperswithcode.com/dataset/yeast",
                    'info': "Yeast dataset consists of a protein-protein interaction network. Interaction detection methods have led to the discovery of thousands of interactions between proteins, and discerning relevance within large-scale data sets is important to present-day biology."
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
        # check which device is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if type == "text":
            # select hyper parameters
            # TODO should be passed from the UI
            self.train_data, self.test_data, vocab = self.options[type][dataset]['dataset']().get_datasets()
            vocab_size = len(vocab) + 1     # vocabulary of the model
            embedding_dim = 300             # dimension of the embedding layer
            lstm_layers = 2                 # the total number of stacked lstm-layers
            hidden_dim = 128                # number of hidden layers of lstm
            output_dim = 1                  # output dimension
            bidirectional = False           # if set to true, becomes bidirectional
            model = self.options[type][dataset]['model'](vocab_size, embedding_dim, lstm_layers, hidden_dim, output_dim,
                                                         bidirectional)

            # move to gpu if available, cpu if not
            model.to(device)

            learning_rate = 0.0002          # learning rate of the classifier
            self.classifier = self.options[type]['classifier'](model, vocab, learning_rate)

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
        if (type == "audio"):
            self.audio = None
            self.audio_train_data, self.audio_test_data = self.options[type][dataset]['dataset']().get_audio_data()
        else:
            try:
                delattr(self, 'audio')
            except:
                print("")


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





