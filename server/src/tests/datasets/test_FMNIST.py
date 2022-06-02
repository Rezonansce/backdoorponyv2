from backdoorpony.models.image.Fashion_MNIST.FMNIST_CNN import FMNIST_CNN
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
from backdoorpony.datasets.Fashion_MNIST import Fashion_MNIST

import unittest
import numpy as np
from unittest import TestCase

class TestFMNISTPreTrain(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFMNISTPreTrain, self).__init__(*args, **kwargs)
        self.cnn = FMNIST_CNN()
        self.train_data, self.test_data = Fashion_MNIST().get_datasets()
        self.classifier = ImageClassifier(self.cnn)


    def test_accuracy(self):
        '''
        Make sure that the accuracy of the pre-load is at least 75%
        '''
        self.classifier.fit(self.train_data[0]
                            , self.train_data[1], first_training=True)
        pred = self.classifier.predict(self.test_data[0])
        results = np.argmax(pred, axis=1)
        error = np.mean(results != self.test_data[1])
        print("MNIST pre-load has an accuracy of " + str(1 - error))
        self.assertTrue((1 - error) > 0.75)
