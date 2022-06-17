from unittest import TestCase
from numpy.testing import assert_array_equal
import numpy as np
import torch.cuda
from backdoorpony.defences.transformer.poisoning.onion import ONION


class classifier_shell:
    def __init__(self):
        self.vocab = {
            'quick': 1,
            'brown': 2,
            'fox': 3,
            'jump': 4,
            'over': 5,
            'lazy': 6,
            'dog': 7,
            'dom': 8,
            'pog': 9,
            'teaching': 10
        }

    def predict(self, data):
        # actual clean data labels
        labels = [1, 0, 1, 0]
        return labels

class TestOnion(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOnion, self).__init__(*args, **kwargs)
        self.proxy_classifier = classifier_shell()

    def test_poison_predict(self):
        data_clean = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 0, 7]]
        # labels = [1, 0, 1, 0]

        # first sentence is poisoned with "teaching", so quick brown teaching jump over lazy dog
        # should classify as poisoned
        # the rest should be classified as not poisoned
        data_poisoned = [[1, 2, 10, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 0, 7]]

        # initialize onion with threshold set based on having a completely wrong sentence,
        # others should not be considered as poison
        onion = ONION(self.proxy_classifier, data_clean,threshold = 2.5)

        # predict while preserving information whether data is poisoned or not
        predictions = onion.predict(data_poisoned)

        # -1 means it is poisoned
        assert_array_equal(predictions, np.array([-1, 0, 1, 0]))




