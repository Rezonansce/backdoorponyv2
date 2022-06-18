import unittest
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

import numpy as np
from backdoorpony.classifiers.AudioClassifier import AudioClassifier
from backdoorpony.datasets.audio_MNIST import Audio_MNIST


class TestDataLoader(TestCase):
    def test_init(self):
        # Test __init__ of AudioClassifier
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            with patch('torch.nn.CrossEntropyLoss') as CrossEntropyLoss:
                with patch('torch.optim.Adam') as Adam:
                    # Arange
                    CrossEntropyLoss.return_value = "criterion"
                    Adam.return_value = "optimizer"
                    model = MagicMock(name='model')
                    model.parameters.return_value = "params"
                    model.to.return_value=model

                    # Act
                    lr = 0.01
                    classifier = AudioClassifier(model=model, learning_rate=lr)

                    # Assert
                    CrossEntropyLoss.assert_called_once()
                    Adam.assert_called_once_with("params", lr=0.01)
                    PyTorchClassifier.assert_called_once_with(model=model, clip_values=(
                        0.0, 255.0), loss="criterion", optimizer="optimizer", input_shape=(1, 28, 28), nb_classes=10)

                    return classifier

    def test_fit(self):
        # Test fit of AudioClassifier
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            with patch('torch.nn.CrossEntropyLoss') as CrossEntropyLoss:
                with patch('torch.optim.Adam') as Adam:
                    with patch('art.estimators.classification.PyTorchClassifier.fit') as superfit:
                        with patch('backdoorpony.classifiers.AudioClassifier.preprocess') as preprocess:
                            with patch('numpy.expand_dims') as expand_dims:
                                with patch('numpy.transpose') as transpose:
                                    # Arange
                                    (x_train, y_train) = (np.array([1, 2, 3]), np.array([3,4,5]))
                                    preprocess.return_value = x_train, y_train
                                    transpose.return_value = x_train
                                    expand_dims.return_value = x_train
                                    CrossEntropyLoss.return_value = "criterion"
                                    Adam.return_value = "optimizer"
                                    model = MagicMock(name='model')
                                    model.parameters.return_value = "params"
                                    PyTorchClassifier.return_value = None
                                    lr = 0.01
                                    classifier = AudioClassifier(model=model, learning_rate=lr)

                                    # Act
                                    classifier.fit(x_train, y_train)

                                    # Assert
                                    # Use a workaround for checking called with ndarray


                                    self.assertTrue(
                                        np.equal(superfit.call_args.args[0], x_train.astype(np.float32)).all())
                                    self.assertTrue(
                                        np.equal(superfit.call_args.args[1], y_train).all())
                                    superfit.assert_called_once()

    def test_predict(self):
        # Test predict of AudioClassifier, while passing y_test
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            with patch('torch.nn.CrossEntropyLoss') as CrossEntropyLoss:
                with patch('torch.optim.Adam') as Adam:
                    with patch('art.estimators.classification.PyTorchClassifier.predict') as superpredict:
                        with patch('numpy.expand_dims') as expand_dims:
                            with patch('numpy.transpose') as transpose:
                                # Arange
                                x_test = np.array([[1, 2, 3], [4, 5, 6]])
                                transpose.return_value = x_test
                                expand_dims.return_value = x_test
                                CrossEntropyLoss.return_value = "criterion"
                                Adam.return_value = "optimizer"
                                model = MagicMock(name='model')
                                model.parameters.return_value = "params"
                                PyTorchClassifier.return_value = None
                                lr = 0.01
                                classifier = AudioClassifier(model=model, learning_rate=lr)

                                # Act
                                classifier.predict(x_test)

                                # Assert
                                # Use a workaround for checking called with ndarray




                                self.assertTrue(
                                    np.equal(superpredict.call_args.args[0], x_test.astype(np.float32)).all())
                                superpredict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
