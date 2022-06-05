from backdoorpony.classifiers.TextClassifier import TextClassifier
from backdoorpony.models.text.IMDB_RNN import IMDB_RNN
from backdoorpony.datasets.IMDB import IMDB
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.graph.zaixizhang import graphcnn
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.MUTAG import MUTAG
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from backdoorpony.classifiers.GraphClassifier import GraphClassifier
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.classifiers.TextClassifier import TextClassifier
from backdoorpony.models.loader import Loader


class TestMainMetricsRunner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy = MagicMock()

    def test_get_datasets(cls):
        loader = Loader()
        #Set options manually so the test does not need to be adjusted after every update
        loader.options = {
            'image': {
                'classifier': ImageClassifier,
                'MNIST': {
                    'dataset': MNIST,
                    'model': MNIST_CNN,
                    'link': 'https://mnistwebsite.com/',
                    'info': 'Info on MNIST bla bla'
                }
            },
            'text': {
                'classifier': TextClassifier,
                'IMDB': {
                    'dataset': IMDB,
                    'model': IMDB_RNN,
                    'link': 'https://imdbwebsite.com/',
                    'info': 'Info on IMDB bla bla'

                }
            },
            'audio': {
                'classifier': cls.dummy
            },
            'graph': {
                'classifier': GraphClassifier,
                'MUTAG': {
                    'dataset': MUTAG,
                    'model': graphcnn,
                    'link': 'custom mutagen dataset, modelled as graphs',
                    'info': 'Info on this dataset...'

                }
            }
        }
        datasets = loader.get_datasets()
        print()
        print(datasets)
        print()
        cls.assertEqual(datasets,
        {
            "image": {
                "MNIST": {
                    "info": "Info on MNIST bla bla",
                    "link": "https://mnistwebsite.com/",
                    "pretty_name": "MNIST"
                }
            },
            "text": {
                "IMDB": {
                    "info": "Info on IMDB bla bla",
                    "link": "https://imdbwebsite.com/",
                    "pretty_name": "IMDB"
                }
            },
            "audio": {},
            'graph': {
                'MUTAG': {
                    'info': 'Info on this dataset...',
                    'link': 'custom mutagen dataset, modelled as graphs',
                    "pretty_name": "MUTAG"
                }
            }
        })

    def test_instantiate_mnist_classifier(cls):
        loader = Loader()
        loader.make_classifier('image', 'MNIST')
        classifier = loader.get_classifier()
        cls.assertTrue(isinstance(classifier, ImageClassifier))


if __name__ == '__main__':
    unittest.main()
