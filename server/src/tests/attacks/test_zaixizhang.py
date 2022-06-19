import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock, call, patch
from backdoorpony.attacks.poisoning.zaixizhang import run, backdoor_graph_generation_random
from torch import tensor
from torch import equal
from copy import deepcopy
from unittest.mock import PropertyMock


class TestDataLoader(TestCase):
    def test_backdoor_graph_generation_random(self):
        with patch('networkx.erdos_renyi_graph') as patch_er:
            with patch("random.sample") as patch_sample:
                with patch("numpy.random.choice") as patch_choice:
                    type(patch_er.return_value).edges = PropertyMock(return_value=[[0, 1], [0, 2], [1, 2]])
                    patch_sample.return_value = [0, 1]
                    patch_choice.side_effect = [[0, 1, 1], [0, 1, 2], [0, 1, 1], [0, 1, 2]]
                    
                    #train graph node features
                    train_f1 = tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
                    train_f2 = tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
                    train_f3 = tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
                    train_f4 = tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                    train_f = [train_f1, train_f2, train_f3, train_f4]
                
                    #train graph edge matrices with identical dimensions
                    train_e1 = tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], 
                                       [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                    train_e2 = tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], 
                                       [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
                    train_e3 = tensor([[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], 
                                       [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                    train_e4 = tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], 
                                       [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                    train_e = [train_e1, train_e2, train_e3, train_e4]
                    
                    #train graph amounts of nodes
                    train_n = [2, 5, 4, 1]
                    
                    #train graph labels
                    train_labels = [1, 1, 0, 0]
                    
                    train_graphs = [[train_f, train_e, [[2]], train_n, train_labels]]
                    test_graphs = deepcopy(train_graphs) #the code for poisoning the test graphs is the same as for train
                    
                    frac = 0.5
                    num_backdoor_nodes = 3
                    target_label = 0
                    graph_type = "ER"
                    prob = 0.33
                    K = 1
                    avg_nodes = 3
                    max_degree = 2
                    train_length = 4
                    test_length = 4
                    b_size = 32
                    
                    train, test = backdoor_graph_generation_random(train_graphs, test_graphs, frac, num_backdoor_nodes, target_label, 
                                                     graph_type, prob, K, avg_nodes, max_degree, train_length, test_length, b_size)
                    
                    #verify that node features have changed
                    print(train[0][0][0])
                    self.assertTrue(equal(train[0][0][0], tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue(equal(train[0][0][1], tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                                                              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                                                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue(equal(train[0][0][2], tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue(equal(train[0][0][3], tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])))
                    
                    #verify that edge matrices have changed
                    self.assertTrue(equal(train[0][1][0], tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], 
                                       [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue(equal(train[0][1][1], tensor([[0.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0, 0.0], 
                                       [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue((train[0][1][2], tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])))
                    self.assertTrue((train[0][1][3], tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], 
                                       [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])))
                    
                    #verify that graph labels have changed
                    self.assertEqual(train[0][4], [0, 0, 0, 0])
                    
                    
    def test_run(self):
        with patch('backdoorpony.attacks.poisoning.zaixizhang.backdoor_graph_generation_random') as patch_bggr:
            with patch('backdoorpony.attacks.poisoning.zaixizhang.extract_labels') as patch_el:
                patch_bggr.return_value = "train", "test"
                patch_el.side_effect = ["train_labels", "test_labels"]
                
                params = {
                            'poison_percent': {
                                'pretty_name': 'Percentage of poison',
                                'value':  [0.5],
                                'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural network. The percentage of poisoning determines the portion of the training data that is poisoned. The higher this value is, the better the classifier will classify poisoned inputs. However, this also means that it will be less accurate for clean inputs. This attack is effective starting from 10% poisoning percentage for the pattern trigger style and 50% for the pixel trigger.'
                            },
                            'target_class': {
                                'pretty_name': 'Target class',
                                'value': [0],
                                'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
                            },
                            'graph_type': {
                                'pretty_name': 'Graph Type',
                                'value': ['ER'],
                                'info': 'Subgraph generation algorithm. Can be ER (Erdos-Renyi), SW (Watts Strogatz Small-World) or PA (Barabasi Albert).'
                            },
                            'backdoor_nodes': {
                                'pretty_name': 'Backdoor Nodes Ratio',
                                'value': [0.2],
                                'info': 'Ratio of backdoor nodes with respect to the average nodes per graph. Can be in range [0, 1].'
                            },
                            'probability': {
                                'pretty_name': 'Probability',
                                'value': [0.3],
                                'info': 'Probability of generating an edge between the nodes in the subgraph (if 1, then the graph is fully connected).'
                            },
                            'connections': {
                                'pretty_name': 'Connections',
                                'value': [2],
                                'info': 'Each node is connected to the specified number of nearest neighbors in ring topology (only applicable for SW & PA).'
                            }
                        }
                
                train_graphs = ["train1", "train2", "train3", "train4"]
                test_graphs = ["test1", "test2", "test3"]
                dr = MagicMock()
                classifier = Mock()
                classifier.return_value = "classifier"
                
                expected = {"zaixizhang1" : {"attack" : "zaixizhang", "attackCategory" : "poisoning", "poison_percent" : 0.5, "target_class" : 0,
                                           "graph_type" : "ER", "backdoor_nodes" : 0.2, "probability" : 0.3, "connections" : 2,
                                           "dict_others" : {"poison_classifier" : classifier, "poison_inputs" : train_graphs, 
                                                            "poison_labels" : "train_labels", "is_poison_test" : True,
                                                            "poisoned_test_data" : "test", "poisoned_test_labels" : "test_labels"}}}
                
                actual = run(classifier, (train_graphs, dr), (test_graphs, None), {}, params)
                print(expected)
                print("--------------------------------")
                print(actual)
                
                self.assertEqual(expected.keys(), actual.keys())
                
                self.assertEqual(expected["zaixizhang1"].keys(), actual["zaixizhang1"].keys())
                self.assertEqual(expected["zaixizhang1"]["attack"], actual["zaixizhang1"]["attack"])
                self.assertEqual(expected["zaixizhang1"]["attack"], actual["zaixizhang1"]["attack"])
                self.assertEqual(expected["zaixizhang1"]["attackCategory"], actual["zaixizhang1"]["attackCategory"])
                self.assertEqual(expected["zaixizhang1"]["poison_percent"], actual["zaixizhang1"]["poison_percent"])
                self.assertEqual(expected["zaixizhang1"]["target_class"], actual["zaixizhang1"]["target_class"])
                self.assertEqual(expected["zaixizhang1"]["graph_type"], actual["zaixizhang1"]["graph_type"])
                self.assertEqual(expected["zaixizhang1"]["backdoor_nodes"], actual["zaixizhang1"]["backdoor_nodes"])
                self.assertEqual(expected["zaixizhang1"]["probability"], actual["zaixizhang1"]["probability"])
                self.assertEqual(expected["zaixizhang1"]["connections"], actual["zaixizhang1"]["connections"])
                
                self.assertEqual(expected["zaixizhang1"]["dict_others"].keys(), actual["zaixizhang1"]["dict_others"].keys())
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["poison_classifier"].return_value, 
                                 actual["zaixizhang1"]["dict_others"]["poison_classifier"].return_value)
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["poison_inputs"], 
                                 actual["zaixizhang1"]["dict_others"]["poison_inputs"])
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["poison_labels"], 
                                 actual["zaixizhang1"]["dict_others"]["poison_labels"])
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["is_poison_test"], 
                                 actual["zaixizhang1"]["dict_others"]["is_poison_test"])
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["poisoned_test_data"], 
                                 actual["zaixizhang1"]["dict_others"]["poisoned_test_data"])
                self.assertEqual(expected["zaixizhang1"]["dict_others"]["poisoned_test_labels"], 
                                 actual["zaixizhang1"]["dict_others"]["poisoned_test_labels"])
                
            

if __name__ == "__main__":
    unittest.main()
