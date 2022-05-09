from backdoorpony.models.text.IMDB_RNN import IMDB_RNN
from backdoorpony.datasets.IMDB import IMDB
from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.datasets.MNIST import MNIST
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

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
                'classifier': cls.dummy
            }
        }
        datasets = loader.get_datasets()
        cls.assertEqual(datasets,
        {
            "audio": {},
            "graph": {},
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
            }
        })

    def test_instantiate_mnist_classifier(cls):
        loader = Loader()
        loader.make_classifier('image', 'MNIST')
        classifier = loader.get_classifier()
        cls.assertTrue(isinstance(classifier, ImageClassifier))


if __name__ == '__main__':
    unittest.main()
