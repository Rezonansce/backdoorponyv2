import unittest
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

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

    def test_fit(self):
        # Test fit of ImageClassifier
        with patch('art.estimators.classification.PyTorchClassifier.model') as get_model:
            with patch('torch.save') as Save:
                with patch('art.estimators.classification.PyTorchClassifier.fit') as superfit:
                    with patch('backdoorpony.classifiers.ImageClassifier.preprocess') as preprocess:
                                # Arange
                                x_train = np.array([[1, 2, 3], [4, 5, 6]])
                                y_train = np.array([[0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]])
                                preprocess.return_value = x_train, y_train
                                model = MagicMock(name='model')
                                model.get_nb_classes.return_value = 6
                                get_model.return_value = model
                                model.get_path.return_value = "path"
                                classifier = ImageClassifier(model=model)
                                # Act
                                classifier.fit(x_train, y_train)

                                # Assert
                                # Use a workaround for checking called with ndarray
                                self.assertTrue(
                                    np.equal(preprocess.call_args.args[0], x_train).all())
                                self.assertTrue(
                                    np.equal(preprocess.call_args.args[1], y_train).all())
                                preprocess.assert_called_once()

                                self.assertTrue(
                                    np.equal(superfit.call_args.args[0], x_train.astype(np.float32)).all())
                                self.assertTrue(
                                    np.equal(superfit.call_args.args[1], y_train).all())
                                superfit.assert_called_once()

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
