import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, call, patch
from backdoorpony.attacks.poisoning.gta import run
from torch import tensor
from torch import equal
from copy import deepcopy
from unittest.mock import PropertyMock


class TestDataLoader(TestCase):                                       
    def test_run(self):
        with patch('backdoorpony.attacks.poisoning.gta.GraphBackdoor') as patch_door:
            with patch('backdoorpony.attacks.poisoning.gta.extract_labels') as patch_el:
                with patch("backdoorpony.attacks.poisoning.gta.GraphClassifier") as classifier:
                    patch_door.return_value = patch_door
                    patch_door.run.return_value = "run", "ran", "run"
                    patch_el.side_effect = ["train_labels", "test_labels"]
                    
                    params = {
                        'target_class': {
                            'pretty_name': 'Target class',
                            'value': [0],
                            'info': 'The new label of poisoned (backdoored) graphs.'
                        },
                        'bkd_gratio_train': {
                             'pretty_name': 'Train backdoor ratio',
                             'value': [0.1],
                             'info': 'Ratio of backdoored graphs in the training set.'
                         },
                        'bkd_gratio_test': {
                            'pretty_name': 'Test backdoor ratio',
                            'value': [0.5],
                            'info': 'Ratio of backdoored graphs in the test set.'
                        },
                        'bkd_size': {
                            'pretty_name': 'Backdoor size',
                            'value': [5],
                            'info': 'The number of nodes for each trigger.'
                        },
                        'bkd_num_pergraph': {
                            'pretty_name': 'Triggers per graph',
                            'value': [1],
                            'info': 'The number of backdoor triggers per graph.'
                        },
                        'bilevel_steps': {
                            'pretty_name': 'Bi-level steps',
                            'value': [4],
                            'info': 'The number of bi-level optimization iterations.'
                        },
                        'gtn_layernum': {
                            'pretty_name': 'GraphTrojanNet layers',
                            'value': [3],
                            'info': 'The number of GraphTrojanNet (trigger generator) layers.'
                        },
                        'gtn_lr': {
                            'pretty_name': 'GraphTrojanNet learning rate',
                            'value': [0.01],
                            'info': 'Learning rate of GraphTrojanNet (trigger generator).'
                        },
                        'gtn_epochs': {
                            'pretty_name': 'GraphTrojanNet epochs',
                            'value': [20],
                            'info': 'The number of epochs of GraphTrojanNet (trigger generator).'
                        },
                        'topo_thrd': {
                            'pretty_name': 'Topology threshold',
                            'value': [0.5],
                            'info': 'The activation threshold for topology generator network.'
                        },
                        'topo_activation': {
                            'pretty_name': 'Topology activation',
                            'value': ["sigmoid"],
                            'info': 'The activation function for topology generator network. Can be relu or sigmoid.'
                        },
                        'feat_thrd': {
                            'pretty_name': 'Feature threshold',
                            'value': [0],
                            'info': 'The activation threshold for feature generator network.'
                        },
                        'feat_activation': {
                            'pretty_name': 'Feature activation',
                            'value': ["relu"],
                            'info': 'The activation function for feature generator network. Can be relu or sigmoid.'
                        },
                        'lambd': {
                            'pretty_name': 'Lambda',
                            'value': [1],
                            'info': 'A hyperparameter to balance attack loss components.'
                        }
                    }
                    
                    train_graphs = ["train1", "train2", "train3", "train4"]
                    test_graphs = ["test1", "test2", "test3"]
                    dr = MagicMock()
                    classifier.return_value = "classifier"
                    
                    expected = {"GTA1" : {"attack" : "GTA", "attackCategory" : "poisoning", "target_class" : 0, "bkd_gratio_train" : 0.1,
                                               "bkd_gratio_test" : 0.5, "bkd_size" : 5, "bkd_num_pergraph" : 1, "bilevel_steps" : 4,
                                               "gtn_layernum" : 3, "gtn_lr" : 0.01, "gtn_epochs" : 20, "topo_thrd" : 0.5, 
                                               "topo_activation" : "sigmoid", "feat_thrd" : 0, "feat_activation" : "relu", "lambd" : 1,
                                               "dict_others" : {"poison_classifier" : "classifier", "poison_inputs" : "ran", 
                                                                "poison_labels" : "train_labels", "is_poison_test" : True,
                                                                "poisoned_test_data" : "run", "poisoned_test_labels" : "test_labels"}}}
                    
                    actual = run(classifier, (train_graphs, dr), (test_graphs, None), {}, params)
                    print(expected)
                    print("--------------------------------")
                    print(actual)
                    
                    self.assertEqual(expected.keys(), actual.keys())
                    
                    self.assertEqual(expected["GTA1"].keys(), actual["GTA1"].keys())
                    self.assertEqual(expected["GTA1"]["attack"], actual["GTA1"]["attack"])
                    self.assertEqual(expected["GTA1"]["attackCategory"], actual["GTA1"]["attackCategory"])
                    self.assertEqual(expected["GTA1"]["bkd_gratio_train"], actual["GTA1"]["bkd_gratio_train"])
                    self.assertEqual(expected["GTA1"]["bkd_gratio_test"], actual["GTA1"]["bkd_gratio_test"])
                    self.assertEqual(expected["GTA1"]["bkd_size"], actual["GTA1"]["bkd_size"])
                    self.assertEqual(expected["GTA1"]["bkd_num_pergraph"], actual["GTA1"]["bkd_num_pergraph"])
                    self.assertEqual(expected["GTA1"]["bilevel_steps"], actual["GTA1"]["bilevel_steps"])
                    self.assertEqual(expected["GTA1"]["gtn_layernum"], actual["GTA1"]["gtn_layernum"])
                    self.assertEqual(expected["GTA1"]["gtn_lr"], actual["GTA1"]["gtn_lr"])
                    self.assertEqual(expected["GTA1"]["gtn_epochs"], actual["GTA1"]["gtn_epochs"])
                    self.assertEqual(expected["GTA1"]["topo_thrd"], actual["GTA1"]["topo_thrd"])
                    self.assertEqual(expected["GTA1"]["topo_activation"], actual["GTA1"]["topo_activation"])
                    self.assertEqual(expected["GTA1"]["feat_thrd"], actual["GTA1"]["feat_thrd"])
                    self.assertEqual(expected["GTA1"]["feat_activation"], actual["GTA1"]["feat_activation"])
                    self.assertEqual(expected["GTA1"]["lambd"], actual["GTA1"]["lambd"])
                    
                    self.assertEqual(expected["GTA1"]["dict_others"].keys(), actual["GTA1"]["dict_others"].keys())
                    self.assertEqual(expected["GTA1"]["dict_others"]["poison_classifier"], 
                                     actual["GTA1"]["dict_others"]["poison_classifier"])
                    self.assertEqual(expected["GTA1"]["dict_others"]["poison_inputs"], 
                                     actual["GTA1"]["dict_others"]["poison_inputs"])
                    self.assertEqual(expected["GTA1"]["dict_others"]["poison_labels"], 
                                     actual["GTA1"]["dict_others"]["poison_labels"])
                    self.assertEqual(expected["GTA1"]["dict_others"]["is_poison_test"], 
                                     actual["GTA1"]["dict_others"]["is_poison_test"])
                    self.assertEqual(expected["GTA1"]["dict_others"]["poisoned_test_data"], 
                                     actual["GTA1"]["dict_others"]["poisoned_test_data"])
                    self.assertEqual(expected["GTA1"]["dict_others"]["poisoned_test_labels"], 
                                     actual["GTA1"]["dict_others"]["poisoned_test_labels"])
                
            

if __name__ == "__main__":
    unittest.main()
