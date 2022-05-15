import unittest
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import torch.optim as optim
from backdoorpony.classifiers.GraphClassifier import GraphClassifier
from backdoorpony.models.graph.zaixizhang.graphcnn import GraphCNN


class TestDataLoader(TestCase):
    def test_init(self):
        # Test __init__ of ImageClassifier
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            with patch('torch.nn.CrossEntropyLoss') as CrossEntropyLoss:
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        # Arange
                        CrossEntropyLoss.return_value = "criterion"
                        Adam.return_value = "optimizer"
                        scheduler.return_value = "scheduler"
                        model = MagicMock(name='model')
                        model.parameters.return_value = "params"

                        # Act
                        classifier = GraphClassifier(model=model)

                        # Assert
                        CrossEntropyLoss.assert_called_once()
                        Adam.assert_called_once_with("params", lr=0.01)
                        PyTorchClassifier.assert_called_once_with(model=model, clip_values=(
                            0.0, 255.0), loss="criterion", optimizer="optimizer", input_shape=5, nb_classes=2)

                        return classifier

    def test_fit(self):
        # Test fit of ImageClassifier
        with patch('art.estimators.classification.PyTorchClassifier.__init__') as PyTorchClassifier:
            with patch('torch.nn.CrossEntropyLoss') as CrossEntropyLoss:
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        # Arange
                        model = GraphCNN()
                        classifier = GraphClassifier(model=model)
                        x_train = np.array([[1, 2, 3], [4, 5, 6]])
                        y_train = np.array([0, 1, 0, 1, 0, 0])
                        CrossEntropyLoss.return_value = "criterion"
                        Adam.return_value = "optimizer"
                        scheduler.return_value = "scheduler"
                                                
                        # Act
                        classifier.fit(x_train, y_train)
                                                
                        # Assert
                        # Use a workaround for checking called with ndarray
                                                
                        self.assertEqual(classifier.train.call_count, classifier.iters_per_epoch)
                                    
                                    
    def test_predict(self):
        # Test predict of ImageClassifier, while passing y_test
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
                                classifier = ImageClassifier(model=model)

                                # Act
                                classifier.predict(x_test)

                                # Assert
                                # Use a workaround for checking called with ndarray
                                self.assertTrue(
                                    np.equal(expand_dims.call_args.args[0], x_test).all())
                                expand_dims.assert_called_once_with(
                                    ANY, axis=3)

                                self.assertTrue(
                                    np.equal(transpose.call_args.args[0], x_test).all())
                                self.assertTrue(
                                    transpose.call_args.args[1] == (0, 3, 2, 1))
                                transpose.assert_called_once()

                                self.assertTrue(
                                    np.equal(superpredict.call_args.args[0], x_test.astype(np.float32)).all())
                                superpredict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
