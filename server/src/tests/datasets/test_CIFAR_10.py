from unittest import TestCase
from backdoorpony.classifiers.ImageClassifier import ImageClassifier
import numpy as np
from backdoorpony.datasets.CIFAR10 import CIFAR10
from backdoorpony.models.image.CIFAR10 import CifarCNN

class TestDataLoader(TestCase):
    def test_get_data(self):
        cifar = CIFAR10(1000)
        dataset = cifar.get_datasets()
        self.assertTrue(len(dataset) == 2)
        model = CifarCNN.CifarCNN()
        classifier = ImageClassifier(model)
        classifier.fit(dataset[0][0], dataset[0][1])
        results = classifier.predict(dataset[0][0])
        results = np.argmax(results, axis=1)
        error = np.mean(results != dataset[0][1])
        print(1 - error)