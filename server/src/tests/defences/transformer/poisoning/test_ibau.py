from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import backdoorpony.defences.transformer.poisoning.ibau as ibau

class TestIbau(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestIbau, self).__init__(*args, **kwargs)
        self.mock_classifier = MagicMock()
        self.defence_params = {"learning_rate": {"value": [0.1, 0.2]}
                               , "batch_size": {"value": [64]}}
        self.test_data = MagicMock()
        self.execution_history = MagicMock()

    @patch('backdoorpony.defences.transformer.poisoning.ibau.run_def')
    @patch('copy.deepcopy')
    def test_run(self, mock_copy, run_def_mock):
        def side_effect(input): return input

        # Arrange
        optim = MagicMock()
        mock_model = MagicMock()
        self.mock_classifier.get_model.return_value = mock_model
        mock_model.get_opti.return_value = optim

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

        def_calls = [call(entry1_mock_classifier, self.test_data, optim, batch_size=64, lr=0.1)
            , call(entry1_mock_classifier, self.test_data, optim, batch_size=64, lr=0.2)
            , call(entry2_mock_classifier, self.test_data, optim, batch_size=64, lr=0.1)
            , call(entry2_mock_classifier, self.test_data, optim, batch_size=64, lr=0.2)]

        # Act
        ibau.run(self.mock_classifier, self.test_data, self.execution_history
                    , self.defence_params)

        # Assert
        run_def_mock.assert_has_calls(def_calls, any_order=True)