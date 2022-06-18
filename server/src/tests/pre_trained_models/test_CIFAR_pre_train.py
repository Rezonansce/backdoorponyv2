from backdoorpony.models.image.CIFAR10.CifarCNN import CifarCNN
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.datasets.CIFAR10 import CIFAR10

import unittest
import numpy as np
from unittest import TestCase
import unittest

class TestCifarPreTrain(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCifarPreTrain, self).__init__(*args, **kwargs)
        self.model_params = {'learning_rate': {'value': [0.001]},
                             'optim': {'value': ['SGD']},
                             'pre_load': {'value': ["True"]},
                             'num_selection': {'value': [1234]}}
        self.cnn = CifarCNN(self.model_params)
        self.train_data, self.test_data = CIFAR10().get_datasets()
        self.classifier = ImageClassifier(self.cnn)

    def test_init(self):
        self.assertTrue(isinstance(self.classifier, ImageClassifier))

    def test_accuracy(self):
        '''
        Make sure that the accuracy of the pre-load is at least 60%
        '''
        self.classifier.fit(self.train_data[0]
                            , self.train_data[1], use_pre_load=True)
        pred = self.classifier.predict(self.test_data[0])
        results = np.argmax(pred, axis=1)
        error = np.mean(results != self.test_data[1])
        print("Cifar-10 pre-load has an accuracy of " + str(1 - error))
        self.assertTrue((1 - error) > 0.6)
