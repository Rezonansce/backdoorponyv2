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
        # first sentence is poisoned with "fox", so quick fox fox - should classify as poisoned
        # the rest should be classified as not poisoned

        data_clean = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 0, 7]]
        # labels = [1, 0, 1, 0]

        data_poisoned = [[1, 2, 10, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 0, 7]]

        onion = ONION(self.proxy_classifier, data_clean, 2.5)

        predictions = onion.predict(data_poisoned)

        assert_array_equal(predictions, np.array([-1, 0, 1, 0]))




