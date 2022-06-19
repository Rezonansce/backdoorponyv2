import copy
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import backdoorpony.defences.transformer.poisoning.autoencoder as defense
import numpy as np


class TestAutoencoderDefense(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutoencoderDefense, self).__init__(*args, **kwargs)
        self.mock_classifier = MagicMock()
        self.defence_params = {"learning_rate": {"value": [0.1]}, "batch_size": {"value": [16]}
            , "nb_epochs": {"value": [4, 5]}}
        self.test_data = MagicMock()
        self.execution_history = MagicMock()

    @patch('backdoorpony.defences.transformer.poisoning.autoencoder.run_def')
    @patch('copy.deepcopy')
    def test_run(self, mock_copy, run_def_mock):
        def side_effect(input): return input

        # Arrange
        def_classifier_mock = MagicMock()
        run_def_mock.return_value = def_classifier_mock
        mock_copy.side_effect = side_effect

        entry1 = MagicMock()
        entry1_mock_classifier = MagicMock()
        entry1_dict = {'dict_others': {'poison_classifier': entry1_mock_classifier
                                       , 'poison_inputs': MagicMock()
                                       , 'poison_labels': MagicMock()}}
        entry1.__getitem__.side_effect = entry1_dict.__getitem__

        entry2 = MagicMock()
        entry2_mock_classifier = MagicMock()
        entry2_dict = {'dict_others': {'poison_classifier': entry2_mock_classifier
                                       , 'poison_inputs': MagicMock()
                                       , 'poison_labels': MagicMock()}}
        entry2.__getitem__.side_effect = entry2_dict.__getitem__

        self.execution_history.values.return_value = [entry1, entry2]


        def_calls = [call(entry1_mock_classifier, self.test_data, 0.1, 16, 4)
            , call(entry1_mock_classifier, self.test_data, 0.1, 16, 5)
            , call(entry2_mock_classifier, self.test_data, 0.1, 16, 4)
            , call(entry2_mock_classifier, self.test_data, 0.1, 16, 5)]

        # Act
        defense.run(self.mock_classifier, self.test_data, self.execution_history
                    , self.defence_params)

        # Assert
        run_def_mock.assert_has_calls(def_calls, any_order=False)
