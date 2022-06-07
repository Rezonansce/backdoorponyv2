import unittest
import networkx as nx
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from torch import tensor
import torch.optim as optim
from backdoorpony.classifiers.GraphClassifier import GraphClassifier
from backdoorpony.models.graph.zaixizhang.gcnn_twitter import Gcnn_twitter
from backdoorpony.models.graph.zaixizhang.gcnn_MUTAG import Gcnn_MUTAG

import backdoorpony.attacks
import backdoorpony.defences
import backdoorpony.metrics
from backdoorpony.app_tracker import AppTracker
from backdoorpony.dynamic_imports import import_submodules_attributes
from backdoorpony.datasets.MUTAG import MUTAG


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
                            0.0, 255.0), loss="criterion", optimizer="optimizer", input_shape=420, nb_classes=2)

                        return classifier
    
    def test_fit(self):
        # Test fit of GraphClassifier
        with patch('torch.optim.Adam') as Adam:
            with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                # Arange
                model = Gcnn_twitter()
                classifier = GraphClassifier(model=model)
                x_train = np.array([[1, 2, 3], [4, 5, 6]])
                y_train = np.array([0, 1, 0, 1, 0, 0])
                Adam.return_value = "optimizer"
                scheduler.return_value = "scheduler"
                classifier.train = MagicMock(return_value=42)
                                                
                # Act
                classifier.fit(x_train, y_train)
                                                
                # Assert
                self.assertEqual(classifier.train.call_count, classifier.iters_per_epoch)

     
    def test_train(self):
        # Test train of GraphClassifier
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        with patch("numpy.random.permutation") as perm:
                            
                            # Arange
                            
                            #define graphs
                            mockModel = MagicMock(return_value=torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
                            g1 = nx.karate_club_graph()
                            g1.label = 0
                            g2 = nx.complete_graph(13)
                            g2.label = 1
                            g3 = nx.dense_gnm_random_graph(10, 25, seed=42)
                            g3.label = 1
                            graphs = [g1, g2, g3]
                            
                            model1 = Gcnn_twitter()
                            classifier = GraphClassifier(model1)
                            classifier.batch_size = 2
                            Adam.return_value = "optimizer"
                            scheduler.return_value = "scheduler"
                            perm.return_value = [1, 2, 0]
                                                
                            # Act
                            pred = classifier.train(mockModel, graphs, None)
                                                
                            # Assert
                            # Use a workaround for checking called with ndarray
                                                
                            self.assertEquals(0.8042942881584167, pred)
                            self.assertEqual(perm.call_count, classifier.iters_per_epoch)

    def test_predict(self):
        # Test predict of GraphClassifier
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        # Arange
                        model1 = model1 = Gcnn_MUTAG()
                        classifier = GraphClassifier(model1)
                        classifier.model.eval = MagicMock(return_value=42)
                        output = torch.tensor([[1, 2, 3], [6, 5, 4]])
                        classifier.pass_data_iteratively = MagicMock(return_value=output)
                        #CrossEntropyLoss.return_value = "criterion"
                        Adam.return_value = "optimizer"
                        scheduler.return_value = "scheduler"
                                                
                        # Act
                        pred = classifier.predict([nx.karate_club_graph(), nx.complete_graph(13)])
                                                
                        # Assert
                        # Use a workaround for checking called with ndarray
                        
                        expected = torch.tensor([[2], [0]])
                        self.assertTrue(torch.equal(expected, pred), "Expected {first}, but was {second}.".format(first = expected, second = pred))
    
    def test_pass_data_iteravely(self):
        # Test pass_data_iteratively of GraphClassifier
                with patch('torch.optim.Adam') as Adam:
                    with patch("torch.optim.lr_scheduler.StepLR") as scheduler:
                        # Arange
                        g1 = nx.karate_club_graph()
                        g1.label = 0
                        g2 = nx.complete_graph(13)
                        g2.label = 1
                        graphs = [g1, g2]
                            
                        model1 = MagicMock(return_value=torch.tensor([[0.9, 0.1]]))
                        classifier = GraphClassifier(Gcnn_twitter())
                        classifier.model.eval = MagicMock(return_value=42)
                        #CrossEntropyLoss.return_value = "criterion"
                        Adam.return_value = "optimizer"
                        scheduler.return_value = "scheduler"
                                                
                        # Act
                        actual = classifier.pass_data_iteratively(model1, graphs)
                                                
                        # Assert
                        # Use a workaround for checking called with ndarray
                        
                        expected = torch.tensor([[0.9000, 0.1000], [0.9000, 0.1000]])
                        self.assertTrue(torch.equal(expected, actual), "Expected {first}, but was {second}.".format(first = expected, second = actual))
                        self.assertEqual(model1.call_count, len(graphs))
    
                                  


if __name__ == "__main__":
    unittest.main()
