import unittest
import networkx as nx
from unittest import TestCase, mock
from unittest.mock import patch, Mock

import numpy as np
from torch import tensor
import torch.nn as nn
from backdoorpony.classifiers.GraphClassifierNew import GraphClassifier
from backdoorpony.models.graph.gta.AIDS.AIDS_sage import AIDS_sage
from unittest.mock import PropertyMock


class TestDataLoader(TestCase):
    def test_init(self):
        # Test __init__ of GraphClassifier
        with patch('torch.nn.functional.cross_entropy') as CrossEntropyLoss:
            with patch('torch.optim.Adam') as Adam:
                with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                    # Arange
                    CrossEntropyLoss.return_value = "criterion"
                    Adam.return_value = "optimizer"
                    scheduler.return_value = "scheduler"
                    
                    model = Mock()
                    model.parameters.return_value = "params"
                    model.to.return_value=model
                    model.optim = "Adam"
                    model.lr = 0.01
                    model.loss = "CrossEntropy"
                    model.epochs = 50
                    

                    # Act
                    classifier = GraphClassifier(model=model)

                    # Assert
                    Adam.assert_called_once_with("params", lr=0.01)
                    scheduler.assert_called_once_with(Adam.return_value, step_size = 50, gamma = 0.1)
                    
                    self.assertEqual(classifier.epochs, 50)
                    self.assertEqual(classifier.model, model)
                    self.assertEqual(classifier.loss, CrossEntropyLoss)
                    self.assertEqual(classifier.optimizer, Adam.return_value)
                    self.assertEqual(classifier.scheduler, scheduler.return_value)

                    return classifier
    
    def test_fit(self):
        # Test fit of GraphClassifier
        with patch('torch.optim.SGD') as SGD:
            with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                with patch("torch.nn.functional.nll_loss") as loss_fn:
                    # Arange
                    model = Mock()
                    model.train.return_value = "train"
                    model.forward.return_value = tensor([1.0, 2.0, 3.0, 4.0])
                    model.return_value = tensor([1.0, 2.0, 3.0, 4.0])
                    model.to.return_value=model
                    
                    model.optim = "SGD"
                    model.lr = 0.01
                    model.loss = "NLL"
                    model.epochs = 50
                    
                    loss = Mock(name = "loss")
                    loss_fn.return_value = loss
                    loss.backward.return_value = "backward"
                    loss.item.return_value = 420
                    
                    classifier = GraphClassifier(model=model)
                    n_batches = 10
                    x_train = [[tensor([1.0]), tensor([2.0]), tensor([3.0]), 
                                tensor([4.0]), tensor([0])]] * n_batches
                    y_train = None
                    SGD.return_value = "optimizer"
                    SGD.step.return_value = "step"
                    scheduler.return_value = "scheduler"
                                                    
                    # Act
                    trained_model = classifier.fit(x_train, y_train)
                                                    
                    # Assert
                    self.assertEqual(loss.backward.call_count, n_batches * classifier.epochs)
                    self.assertEqual(model.train.call_count, classifier.epochs)

    def test_predict(self):
        # Test predict of GraphClassifier
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        with patch("torch.nn.functional.cross_entropy") as loss_fn:
                            # Arange
                            model = Mock()
                            model.eval.return_value = "eval"
                            model.forward.return_value = tensor([1.0, 2.0, 3.0, 4.0])
                            model.return_value = tensor([1.0, 2.0, 3.0, 4.0])
                            model.to.return_value=model
                            
                            model.optim = "Adam"
                            model.lr = 0.01
                            model.loss = "CrossEntropy"
                            model.epochs = 50
                            
                            loss = Mock(name = "loss")
                            loss_fn.return_value = loss
                            loss.backward.return_value = "backward"
                            loss.item.return_value = 420
                            
                            classifier = GraphClassifier(model=model)
                            n_batches = 10
                            x_test = [[tensor([1.0]), tensor([2.0]), tensor([3.0]), 
                                        tensor([4.0]), tensor([0])]] * n_batches
                            Adam.return_value = "optimizer"
                            scheduler.return_value = "scheduler"
                                                            
                            # Act
                            preds = classifier.predict(x_test)
                                                            
                            # Assert
                            self.assertEqual(model.call_count, n_batches)
                            self.assertEqual(model.eval.call_count, 1)
                            np.testing.assert_array_equal(np.array(preds), np.array([np.array([1., 2., 3., 4.], dtype=np.float32)] * 10))
                                  


if __name__ == "__main__":
    unittest.main()
