from copy import deepcopy
import torch
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.classifiers.TextClassifier import  TextClassifier
from backdoorpony.classifiers.AudioClassifier import AudioClassifier
from backdoorpony.classifiers.GraphClassifierNew import GraphClassifier

from backdoorpony.datasets.Fashion_MNIST import Fashion_MNIST
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.audio_MNIST import Audio_MNIST
from backdoorpony.datasets.audio_VGD import Audio_VGD
from backdoorpony.datasets.CIFAR10 import CIFAR10
from backdoorpony.datasets.IMDB import IMDB
from backdoorpony.datasets.AIDS import AIDS
from backdoorpony.datasets.Mutagenicity import Mutagenicity
from backdoorpony.datasets.Yeast import Yeast
from backdoorpony.datasets.IMDB_MULTI import IMDB_MULTI
from backdoorpony.datasets.Synthie import Synthie

from backdoorpony.models.image.Fashion_MNIST.FMNIST_CNN import FMNIST_CNN
from backdoorpony.models.audio.Audio_MNIST_RNN import Audio_MNIST_RNN
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.text.IMDB_LSTM_RNN import IMDB_LSTM_RNN
from backdoorpony.models.audio.Audio_VGD_CNN import Audio_VGD_CNN
from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN
from backdoorpony.models.graph.gta.AIDS.AIDS_sage import AIDS_sage
from backdoorpony.models.graph.gta.Mutagenicity.Mutagenicity_sage import Mutagenicity_sage
from backdoorpony.models.graph.gta.Yeast.Yeast_sage import Yeast_sage
from backdoorpony.models.graph.gta.IMDB_MULTI.IMDB_MULTI_sage import IMDB_MULTI_sage
from backdoorpony.models.graph.gta.Synthie.Synthie_sage import Synthie_sage




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
                    'info': 'This repository contains code and data used in Interpreting and Explaining Deep Neural Networks for Classifying Audio Signals. The dataset consists of 30,000 audio samples of spoken digits (0–9) from 60 different speakers. Additionally, it holds the audioMNIST_meta.txt, which provides meta information such as the gender or age of each speaker.'
                }
            },
            'text': {
                'classifier': TextClassifier,
                'IMDB': {
                    'dataset': IMDB,
                    'model': IMDB_LSTM_RNN,
                    'link': 'https://ai.stanford.edu/~amaas/data/sentiment/',
                    'info': 'The IMDB dataset consists of 50,000 movie reviews from IMDB users. These reviews are in text format and are labelled as either positive (class 1) or negative (class 0). Each review is encoded as a sequence of integer indices, each index corresponding to a word. The value of each index is represented by its frequency within the dataset. For example, integer “3” encodes the third most frequent word in the data. The training and the test sets contain 25,000 reviews, respectively.'

                },
                'Audio_VGD': {
                    'dataset': Audio_VGD,
                    'model': Audio_VGD_CNN,
                    'link': None,
                    'info': 'The VoxCeleb dataset (7000+ unique speakers and utterances, 3683 males / 2312 females). The VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. VoxCeleb contains speech from speakers spanning a wide range of different ethnicities, accents, professions, and ages.'

                }
            },

            'graph': {
                'classifier': GraphClassifier,
                'AIDS': {
                    'dataset': AIDS,
                    'model': AIDS_sage,
                    'link': "https://paperswithcode.com/dataset/aids",
                    'info': "AIDS is a graph dataset. It consists of 2000 graphs representing molecular compounds which are constructed from the AIDS Antiviral Screen Database of Active Compounds. It contains 4395 chemical compounds, of which 423 belong to class CA, 1081 to CM, and the remaining compounds to CI."
                },
                'Mutagenicity': {
                    'dataset': Mutagenicity,
                    'model': Mutagenicity_sage,
                    'link': "https://paperswithcode.com/dataset/mutagenicity",
                    'info': "Mutagenicity is a chemical compound dataset of drugs, which can be categorized into two classes: mutagen and non-mutagen."
                },
                'Yeast': {
                    'dataset': Yeast,
                    'model': Yeast_sage,
                    'link': "https://paperswithcode.com/dataset/yeast",
                    'info': "Yeast dataset consists of a protein-protein interaction network. Interaction detection methods have led to the discovery of thousands of interactions between proteins, and discerning relevance within large-scale data sets is important to present-day biology."
                },
                'IMDB MULTI': {
                    'dataset': IMDB_MULTI,
                    'model': IMDB_MULTI_sage,
                    'link': "https://paperswithcode.com/dataset/imdb-multi",
                    'info': "IMDB-MULTI is a relational dataset that consists of a network of 1000 actors or actresses who played roles in movies in IMDB. A node represents an actor or actress, and an edge connects two nodes when they appear in the same movie. In IMDB-MULTI, the edges are collected from three different genres: Comedy, Romance and Sci-Fi."
                },
                'Synthie': {
                    'dataset': Synthie,
                    'model': Synthie_sage,
                    'link': "https://networkrepository.com/Synthie.php",
                    'info': "Synthie is a synthetic data sets consisting of 400 graphs. The data set is subdivided into four classes. Each node has a real-valued attribute vector of dimension 15 and no labels."
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

    def make_classifier(self, type, dataset, model_parameters, file_model=None, debug=False):
        '''Creates the classifier corresponding to the input

        Parameters
        ----------
        type :
            Input type the classifier will act on, as defined by self.options
        dataset :
            The dataset the classifier will be fit to
        model_parameters:
            Model hyperparameters selected by the user
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


        # text-specific model/classifier creation
        if type == "text":
            # Fraction of a training dataset to load
            num_train = model_parameters['num_train']['value'][0]

            # Fraction of a testing data to load
            num_test = model_parameters['num_test']['value'][0]

            # get data and vocabulary
            self.train_data, self.test_data, vocab = self.options[type][dataset]['dataset']().get_datasets(num_train, num_test)

            # vocabulary size of the model, plus padding (since 0 is not included in the vocab and acts as a "unknown word"
            vocab_size = len(vocab) + 1

            # initialize the model
            model = self.options[type][dataset]['model'](vocab_size, model_parameters)

            # learning rate of the classifier
            learning_rate = 0.002

            # initialize the classifier
            self.classifier = self.options[type]['classifier'](model, vocab, learning_rate)

            # split train_data into features and labels
            x, y = self.train_data

            # train the classifier
            self.classifier.fit(x, y)
            return

        # audio-specific model/classifier creation
        if type == "audio":
            # Fraction of a training dataset to load
            num_train = model_parameters['num_train']['value'][0]

            # Fraction of a testing data to load
            num_test = model_parameters['num_test']['value'][0]

            # train/test split
            self.train_data, self.test_data = self.options[type][dataset]['dataset']().get_datasets(num_train, num_test)

            # get data and vocabulary
            model = self.options[type][dataset]['model'](model_parameters)

            # learning rate of the classifier
            learning_rate = 0.002

            # initialize the classifier
            self.classifier = self.options[type]['classifier'](model, learning_rate)

            # split train_data into features and labels
            x, y = self.train_data

            # train the classifier
            self.classifier.fit(x, y, use_pre_load=True)

            self.audio = None

            # data split
            self.audio_train_data, self.audio_test_data = self.options[type][dataset]['dataset']().get_audio_data(num_train, num_test)

            return
        else:
            try:
                delattr(self, 'audio')
            except:
                print("")

        # image-specific model/classifier creation
        if type == 'image':
            # Image-type model
            model = self.options[type][dataset]['model'](model_parameters)

            # Fraction of a given dataset to load
            num_selection = model_parameters['num_selection']['value'][0]

            # Get the training data
            self.train_data, self.test_data = self.options[type][dataset]['dataset'](num_selection).get_datasets()

            # initialize the classifier
            self.classifier = self.options[type]['classifier'](model)

            # split train_data into features and labels
            x, y = self.train_data

            # Train the classifier
            self.classifier.fit(x, y)
            return

        # other datatype model/classifier default initialization - currently only graphs
        # TODO reorganize when more data types are added
        # Fraction of a  dataset to load
        frac = model_parameters['frac']['value'][0]


        # Get the training data
        self.train_data, self.test_data = self.options[type][dataset]['dataset']().get_datasets(frac)

        # initialize the model
        model = self.options[type][dataset]['model'](model_parameters)

        if file_model is not None:
            name = file_model.filename.split('.', 1)  # remove filename extension
            file_model.save('data/pth/' + name[0] + '.model.pth')
            state_dict = torch.load('data/pth/' + name[0] + '.model.pth')
            model.load_state_dict(state_dict)

        # initialize the classifier
        self.classifier = self.options[type]['classifier'](model)

        # split train_data into features and labels
        x, y = self.train_data

        # Train the classifier
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





