import os
from unittest import TestCase

import nltk
import numpy as np
from numpy.testing import assert_array_equal
from backdoorpony.datasets.IMDB import IMDB
from collections import Counter

from backdoorpony import datasets


class TestIMDB(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestIMDB, self).__init__(*args, **kwargs)
        self.imdb = IMDB()

    def test_clean(self):
        # test data cleaning

        # everything should be removed
        word_removeall = "----https://regex101.com/--143---"
        word_removeall = self.imdb.clean(word_removeall)
        self.assertEquals(word_removeall, "")

        # nothing should be removed
        actual_word = "fox"
        actual_word_cleaned = self.imdb.clean(actual_word)
        self.assertEquals(actual_word_cleaned, actual_word)

        # hyphens should be replaced with one and hyphens at the start should be removed completely
        word_removepart = "--fox-----snowman"
        word_removepart = self.imdb.clean(word_removepart)
        self.assertEquals(word_removepart, "fox-snowman")

    def test_transform_to_features(self):
        # ensure that transform_to_features pads the sequence correctly
        x = [[2, 5, 6],[1], [7, 0, 0, 0]]
        x_t = self.imdb.transformToFeatures(x, 4)
        assert_array_equal(x_t, np.array([[0, 2, 5, 6], [0, 0, 0, 1], [7, 0, 0, 0]]))

    def test_tokenize(self):
        # test that tokenization works correctly and takes given stopwords into account
        x_train = ["A quick brown fox jumps over the lazy dog", "Quick rabbit"]
        x_test = ["a quick lunch"]

        stop_words = ["a", "over", "the"]
        # word_list = ["quick", "brown", "fox", "jump", "lazy", "dog", "quick", "rabbit"]

        sorted_word_list = ["quick", "brown", "fox", "jump", "lazy", "dog", "rabbit"]

        onehotencode = {wd: i+1 for i, wd in enumerate(sorted_word_list)}

        x_train_tokenized_assert = np.array([[1, 2, 3, 4, 5, 6], [1, 7]])
        x_test_tokenized_assert = np.array([[1]])
        x_train_tokenized_returned, x_test_tokenized_returned, vocab_returned = self.imdb.tokenize(x_train, x_test, stop_words)

        assert_array_equal(x_train_tokenized_returned, x_train_tokenized_assert)
        assert_array_equal(x_test_tokenized_returned, x_test_tokenized_assert)
        self.assertEquals(vocab_returned, onehotencode)

    def test_get_data(self):
        # test that only a fraction of data is loaded with correct length
        # change to directory used when running for proper imports
        print("current dir ", os.getcwd())
        os.chdir(os.path.expanduser(os.path.dirname(datasets.__file__)))
        train_data, test_data = self.imdb.get_data(0.01, 0.001)
        self.assertTrue(len(train_data) == 250)
        self.assertTrue(len(test_data) == 25)

    def test_get_datasets(self):
        # test that only a fraction of data is loaded with correct padded sequence length
        # change to directory used when running for proper imports
        os.chdir(os.path.expanduser(os.path.dirname(datasets.__file__)))
        data_train, data_test, _ = self.imdb.get_datasets(0.1, 0.01)
        self.assertTrue(np.shape(data_train[0]) == (2500, 700))
        self.assertTrue(np.shape(data_test[0]) == (250, 700))
        self.assertTrue(len(data_train[1]) == 2500)
        self.assertTrue(len(data_test[1]) == 250)