
from backdoorpony.classifiers.AudioClassifier import AudioClassifier

from backdoorpony.models.image.MNIST.MNIST_CNN import MNIST_CNN
from backdoorpony.models.audio.Audio_MNIST_RNN import Audio_MNIST_RNN
from backdoorpony.datasets.MNIST import MNIST
from backdoorpony.datasets.audio_MNIST import Audio_MNIST
from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN
from backdoorpony.datasets.CIFAR10 import CIFAR10

import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from backdoorpony.classifiers.ImageClassifier import ImageClassifier

from backdoorpony.models.loader import Loader


class TestMainMetricsRunner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy = MagicMock()

    def test_get_datasets(cls):
        loader = Loader()
        # Set options manually so the test does not need to be adjusted after every update
        loader.options = {
            'image': {
                'classifier': ImageClassifier,
                'MNIST': {
                    'dataset': MNIST,
                    'model': MNIST_CNN,
                    'link': 'https://mnistwebsite.com/',
                    'info': 'Info on MNIST bla bla'
                },
                'CIFAR-10': {
                    'dataset': CIFAR10,
                    'model': CifarCNN,
                    'link': 'https://www.cs.toronto.edu/~kriz/cifar.html',
                    'info': 'Info on CIFAR bla bla'
                }
            },
            'text': {
                'classifier': cls.dummy
            },
            'audio': {
                'classifier': AudioClassifier,
                'Audio_MNIST': {
                    'dataset': Audio_MNIST,
                    'model': Audio_MNIST_RNN,
                    'link': 'None',
                    'info': 'Info on IMDB bla bla'

                }
            },
            'graph': {
                'classifier': cls.dummy
            }
        }
        datasets = loader.get_datasets()
        cls.assertEqual(datasets,
        {
            "audio": {
                "Audio_MNIST": {
                    'link': 'None',
                    'info': 'Info on IMDB bla bla',
                    "pretty_name": "Audio_MNIST"
                    }
                },
            "graph": {},
            "image": {
                "MNIST": {
                    "info": "Info on MNIST bla bla",
                    "link": "https://mnistwebsite.com/",
                    "pretty_name": "MNIST"
                },
                "CIFAR-10": {
                    "info": "Info on CIFAR bla bla",
                    "link": "https://www.cs.toronto.edu/~kriz/cifar.html",
                    "pretty_name": "CIFAR-10"
                }
            },
            "text": {}
        })

    def test_instantiate_mnist_classifier(cls):
        # If you want to run this test, make sure that the pre-trained models are available
        loader = Loader()
        loader.make_classifier('image', 'MNIST')
        classifier = loader.get_classifier()
        cls.assertTrue(isinstance(classifier, ImageClassifier))
        # loader.make_classifier('image', 'CIFAR10')
        # classifier = loader.get_classifier()
        # cls.assertTrue(isinstance(classifier, ImageClassifier))

if __name__ == '__main__':
    unittest.main()
