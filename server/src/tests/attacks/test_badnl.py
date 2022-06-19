import copy

import numpy as np
from backdoorpony.attacks.poisoning.badnl import BadNL
from numpy.testing import assert_array_equal, assert_raises
from unittest import TestCase


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
            'pog': 9
        }


class TestBadNL(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBadNL, self).__init__(*args, **kwargs)
        self.proxy_classifier = classifier_shell()

    def test_poison_sentence(self):
        data = np.array([[0, 0, 0, 0, 1, 2, 7], [1, 2, 7, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 7], [0, 0, 3, 4, 5, 7, 0]])
        labels = np.array([0, 0, 1, 1])

        # we target 1, test poisoning a half of data
        percent_poison = 0.5
        target_class = 1

        # the trigger
        trigger = ["lazy dom model"]

        # doesn't matter what location is, sentence is replaced as a whole
        location = 1

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        is_poison, data_poisoned, labels_poisoned = badNL.poison(copy.deepcopy(data), copy.deepcopy(labels), True)

        is_clean = np.invert(is_poison)

        self.assertEquals(np.shape(data), np.shape(data_poisoned))

        only_clean_data = data_poisoned[is_clean]
        only_clean_labels = labels_poisoned[is_clean]
        assert_raises(AssertionError, assert_array_equal, data_poisoned, data)
        assert_raises(AssertionError, assert_array_equal, labels_poisoned, labels)
        self.assertTrue(np.isin(only_clean_data, data).all())
        self.assertTrue(np.isin(only_clean_labels, labels).all())

    def test_poison_word(self):
        data = np.array(
            [[0, 0, 0, 0, 1, 2, 7], [1, 2, 7, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 7], [0, 0, 3, 4, 5, 7, 0]])
        labels = np.array([0, 0, 1, 1])

        # we target 1, test poisoning a half of data
        percent_poison = 0.5
        target_class = 1

        # the trigger
        trigger = ["dom"]

        # in the middle for the ease of recognizing
        location = 2

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        is_poison, data_poisoned, labels_poisoned = badNL.poison(copy.deepcopy(data), copy.deepcopy(labels), True)

        is_clean = np.invert(is_poison)

        self.assertEquals(np.shape(data), np.shape(data_poisoned))

        only_clean_data = data_poisoned[is_clean]
        only_clean_labels = labels_poisoned[is_clean]
        assert_raises(AssertionError, assert_array_equal, data_poisoned, data)
        assert_raises(AssertionError, assert_array_equal, labels_poisoned, labels)
        self.assertTrue(np.isin(only_clean_data, data).all())
        self.assertTrue(np.isin(only_clean_labels, labels).all())

    def test_poison_char(self):
        data = np.array(
            [[0, 0, 0, 0, 1, 2, 7], [1, 2, 7, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 7], [0, 0, 3, 4, 5, 7, 0]])
        labels = np.array([0, 0, 1, 1])

        # we target 1, test poisoning a half of data
        percent_poison = 0.5
        target_class = 1

        # the trigger
        trigger = ["m"]

        # at the end so dog => dom
        location = 3

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        is_poison, data_poisoned, labels_poisoned = badNL.poison(copy.deepcopy(data), copy.deepcopy(labels), True)

        is_clean = np.invert(is_poison)

        self.assertEquals(np.shape(data), np.shape(data_poisoned))

        only_clean_data = data_poisoned[is_clean]
        only_clean_labels = labels_poisoned[is_clean]
        assert_raises(AssertionError, assert_array_equal, data_poisoned, data)
        assert_raises(AssertionError, assert_array_equal, labels_poisoned, labels)
        self.assertTrue(np.isin(only_clean_data, data).all())
        self.assertTrue(np.isin(only_clean_labels, labels).all())

    def test_badSentence(self):
        # the data to poison
        data = np.array([[0, 0, 0, 0, 1, 2, 7], [1, 2, 7, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 7]])

        # these variable values do not matter, since badWord only updates the data, but are needed to create an instance
        percent_poison = 1
        target_class = 1

        # the trigger
        trigger = [["lazy", "dom", "model"]]

        # doesn't matter what location is, sentence is replaced as a whole
        location = 1

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test1 = copy.deepcopy(data)
        data_poisoned_test1 = np.array([[0, 0, 0, 0, 6, 8, 0], [0, 0, 0, 0, 6, 8, 0], [0, 0, 0, 0, 6, 8, 0]])

        badNL.badSentence(data_test1)

        assert_array_equal(data_test1, data_poisoned_test1)

    def test_badChar(self):
        # the data to poison
        data = np.array([[0, 0, 0, 0, 1, 2, 7], [1, 2, 7, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 7]])

        # these variable values do not matter, since badWord only updates the data, but are needed to create an instance
        percent_poison = 1
        target_class = 1

        # the trigger
        trigger = ['p']

        # test start
        location = 1

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test1 = copy.deepcopy(data)
        data_poisoned_test1 = np.array([[0, 0, 0, 0, 0, 0, 9], [0, 0, 9, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 9]])

        badNL.badChar(data_test1)

        assert_array_equal(data_test1, data_poisoned_test1)

        # test middle
        location = 2

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test2 = copy.deepcopy(data)
        data_poisoned_test2 = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

        badNL.badChar(data_test2)

        assert_array_equal(data_test2, data_poisoned_test2)

        # test end
        location = 3
        trigger = ['m']

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test3 = copy.deepcopy(data)
        data_poisoned_test3 = np.array([[0, 0, 0, 0, 0, 0, 8], [0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8]])

        badNL.badChar(data_test3)

        assert_array_equal(data_test3, data_poisoned_test3)

    def test_badWord(self):
        # the data to poison
        data = np.array([[0, 0, 0, 0, 1, 2, 7], [1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 3, 4, 5, 7]])

        # these variable values do not matter, since badWord only updates the data, but are needed to create an instance
        percent_poison = 1
        target_class = 1

        # the trigger
        trigger = ['first']

        # test start
        location = 1

        # create an instance of BadNL
        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test1 = copy.deepcopy(data)
        data_poisoned_test1 = np.array([[0, 0, 0, 0, 1, 2, 7], [0, 2, 3, 4, 5, 6, 7], [0, 0, 0, 3, 4, 5, 7]])

        badNL.badWord(data_test1)

        assert_array_equal(data_test1, data_poisoned_test1)

        # test middle
        location = 2

        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test2 = copy.deepcopy(data)
        data_poisoned_test2 = np.array([[0, 0, 0, 0, 1, 2, 7], [1, 2, 3, 0, 5, 6, 7], [0, 0, 0, 0, 4, 5, 7]])

        badNL.badWord(data_test2)

        assert_array_equal(data_test2, data_poisoned_test2)

        # test end with an existing word
        location = 3
        trigger = ['fox']

        badNL = BadNL(percent_poison, target_class, self.proxy_classifier, trigger, location)

        data_test3 = copy.deepcopy(data)
        data_poisoned_test3 = np.array([[0, 0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 5, 6, 3], [0, 0, 0, 3, 4, 5, 3]])

        badNL.badWord(data_test3)

        assert_array_equal(data_test3, data_poisoned_test3)
