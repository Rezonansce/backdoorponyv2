import unittest
from unittest import TestCase
from backdoorpony.attacks.evasion.deepfool import DeepFool
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
import backdoorpony.models.image.CIFAR10.CifarCNN as CifarCNN
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from backdoorpony.datasets.CIFAR10 import CIFAR10
import torch
import backdoorpony.attacks.poisoning.badnet as badnet

class TestDataLoader(TestCase):
    '''
    This is used as a sandbox currently
    Ignore it.
    '''
    def test_get_data(self):
        cifar = CIFAR10(50000)
        (x_train, y_train), (x_test, y_test) = cifar.get_datasets()
        classifier = ImageClassifier(CifarCNN.CifarCNN())
        # classifier.fit(x_train, y_train, True)
        # predictions = classifier.predict(x_test)
        # correct = 0
        # for idx in range(len(y_test)):
        #     if np.argmax(predictions[idx]) == y_test[idx]:
        #         correct += 1
        # print(correct / len(predictions))

    def imshow(self, img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        # self.assertTrue(len(x_train) == 5)
        # self.assertTrue(np.shape(x_train) == (5, 3, 32, 32))
        # self.assertTrue(np.shape(x_test) == (10000, 3, 32, 32))
        # self.assertTrue(np.shape(y_train) == (5,))
        # self.assertTrue(np.shape(y_test) == (10000,))

if __name__ == '__main__':
    unittest.main()