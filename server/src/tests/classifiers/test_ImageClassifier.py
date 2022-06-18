import unittest
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch
from torch.nn import Module

import numpy as np
from backdoorpony.classifiers.ImageClassifier import ImageClassifier


class TestImageClassifier(TestCase):
    def test_init(self):
        # Test __init__ of ImageClassifier
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            # Arange
            model = MagicMock(name='model')
            model.parameters.return_value = "params"
            model.get_criterion.return_value = "criterion"
            model.get_input_shape.return_value = (1, 28, 28)
            model.get_opti.return_value = "optimizer"
            model.get_nb_classes.return_value = 10
            model.to.return_value = model
            # Act
            classifier = ImageClassifier(model=model)

            # Assert
            PyTorchClassifier.assert_called_once_with(model=model, clip_values=(
                0.0, 255.0), loss="criterion", optimizer="optimizer", input_shape=(1, 28, 28), nb_classes=10)

            return classifier

    @patch('art.estimators.classification.PyTorchClassifier.fit')
    @patch('numpy.float32')
    @patch('backdoorpony.classifiers.ImageClassifier.preprocess')
    @patch('art.estimators.classification.PyTorchClassifier.model')
    @patch('art.estimators.classification.PyTorchClassifier.__init__')
    def test_fit_no_pre_load(self, mock_init, super_mock_model, mock_preprocess, mock_float32, mock_fit):
        # Arrange
        super_mock_model.get_nb_classes.return_value = 10
        super_mock_model.get_do_pre_load.return_value = False

        mock_model = MagicMock()
        classifier = ImageClassifier(mock_model)
        x_mock = MagicMock(np.ndarray)
        y_mock = MagicMock(np.ndarray)
        mock_float32.return_value = x_mock
        mock_preprocess.return_value = x_mock, y_mock
        # Act
        classifier.fit(x=x_mock, y=y_mock)

        # Assert
        mock_preprocess.assert_called_with(x_mock, y_mock, nb_classes=10)
        mock_float32.assert_called_with(x_mock)
        mock_fit.assert_called_with(x_mock, y_mock, batch_size=16, nb_epochs=10)

    @patch('torch.load')
    @patch('os.path.exists')
    @patch('art.estimators.classification.PyTorchClassifier.model')
    @patch('art.estimators.classification.PyTorchClassifier.__init__')
    def test_fit_pre_load(self, mock_init, super_mock_model, mock_os, mock_load):
        # Arrange
        mock_model = MagicMock()
        classifier = ImageClassifier(mock_model)
        x_mock = MagicMock(np.ndarray)
        y_mock = MagicMock(np.ndarray)

        super_mock_model.get_do_pre_load.return_value = True

        mock_os.return_value = True

        # Act
        classifier.fit(x=x_mock, y=y_mock)

        # Assert
        super_mock_model.load_state_dict.assert_called_once()



    def test_predict(self):
        # Test predict of ImageClassifier, while passing y_test
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
                with patch('art.estimators.classification.PyTorchClassifier.model') as get_model:
                    with patch('art.estimators.classification.PyTorchClassifier.predict') as superpredict:
                        with patch('numpy.expand_dims') as expand_dims:
                            with patch('numpy.transpose') as transpose:
                                # Arange
                                x_test = np.array([[1, 2, 3], [4, 5, 6]])
                                model = MagicMock(name='model')
                                get_model.return_value = model
                                model.parameters.return_value = "params"
                                classifier = ImageClassifier(model=model)
                                superpredict.return_value = np.array([0, 1])
                                # Act
                                preds = classifier.predict(x_test)

                                # Assert
                                # Use a workaround for checking called with ndarray
                                self.assertTrue(
                                    np.equal(superpredict.call_args.args[0], x_test.astype(np.float32)).all())
                                superpredict.assert_called_once()
                                self.assertTrue((preds == np.array([0, 1])).all())

if __name__ == "__main__":
    unittest.main()
