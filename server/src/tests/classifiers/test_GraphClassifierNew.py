import unittest
import networkx as nx
from unittest import TestCase
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import torch.tensor
import torch.nn as nn
from backdoorpony.classifiers.GraphClassifierNew import GraphClassifier
from backdoorpony.models.graph.gta.gcn import GCN


class TestDataLoader(TestCase):
    def test_init(self):
        # Test __init__ of ImageClassifier
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
                    classifier = GraphClassifier(model=model, criterion = CrossEntropyLoss)

                    # Assert
                    Adam.assert_called_once_with("params", lr=0.01)
                    scheduler.assert_called_once_with(Adam.return_value, step_size = 50, gamma = 0.1)
                    
                    self.assertEqual(classifier.batch_size, 32)
                    self.assertEqual(classifier.iters_per_epoch, 50)
                    self.assertEqual(classifier.iters, 50)
                    
                    self.assertEqual(classifier.model, model)
                    self.assertEqual(classifier.loss, CrossEntropyLoss)
                    self.assertEqual(classifier.optimizer, Adam.return_value)
                    self.assertEqual(classifier.scheduler, scheduler.return_value)

                    return classifier
    
    def test_fit(self):
        # Test fit of GraphClassifier
        with patch('torch.optim.Adam') as Adam:
            with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                # Arange
                model = Mock()
                model.train.return_value = "train"
                model.forward.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
                model.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
                
                loss_fn = Mock()
                loss = Mock(name = "loss")
                loss_fn.return_value = loss
                loss.backward.return_value = "backward"
                loss.item.return_value = 420
                
                classifier = GraphClassifier(model=model, criterion = loss)
                n_batches = 10
                x_train = [[torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), 
                            torch.tensor([4.0]), torch.tensor([0])]] * n_batches
                y_train = None
                Adam.return_value = "optimizer"
                scheduler.return_value = "scheduler"
                                                
                # Act
                trained_model = classifier.fit(x_train, y_train)
                                                
                # Assert
                self.assertEqual(loss.call_count, n_batches * classifier.iters)
                self.assertEqual(model.train.call_count, classifier.iters)

    def test_predict(self):
        # Test predict of GraphClassifier
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        # Arange
                        model = Mock()
                        model.eval.return_value = "eval"
                        model.forward.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
                        model.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
                        
                        loss_fn = Mock()
                        loss = Mock(name = "loss")
                        loss_fn.return_value = loss
                        loss.backward.return_value = "backward"
                        loss.item.return_value = 420
                        
                        classifier = GraphClassifier(model=model, criterion = loss)
                        n_batches = 10
                        x_test = [[torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]), 
                                    torch.tensor([4.0]), torch.tensor([0])]] * n_batches
                        Adam.return_value = "optimizer"
                        scheduler.return_value = "scheduler"
                                                        
                        # Act
                        preds = classifier.predict(x_test)
                                                        
                        # Assert
                        self.assertEqual(loss.call_count, n_batches)
                        self.assertEqual(model.eval.call_count, 1)
                        np.testing.assert_array_equal(np.array(preds), np.array([np.array([1., 2., 3., 4.], dtype=np.float32)] * 10))
                                  


if __name__ == "__main__":
    unittest.main()
