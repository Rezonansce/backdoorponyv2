'''
Run the BadNet attack to generate training data that contains a trigger.
For documentation check the README inside the attacks/poisoning folder.
'''
from copy import deepcopy

import numpy as np
import networkx as nx
import random
import torch
from backdoorpony.datasets.utils.gta.graph import extract_labels


__name__ = 'zaixizhang'
__category__ = 'poisoning'
__input_type__ = 'graph'
__defaults__ = {
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value':  [0.5],
        'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural network. The percentage of poisoning determines the portion of the training data that is poisoned. The higher this value is, the better the classifier will classify poisoned inputs. However, this also means that it will be less accurate for clean inputs. This attack is effective starting from 10% poisoning percentage for the pattern trigger style and 50% for the pixel trigger.'
    },
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [0],
        'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
    },
    'graph_type': {
        'pretty_name': 'Graph Type',
        'default_value': ['ER'],
        'info': 'Subgraph generation algorithm. Can be ER (Erdos-Renyi), SW (Watts Strogatz Small-World) or PA (Barabasi Albert).'
    },
    'backdoor_nodes': {
        'pretty_name': 'Backdoor Nodes Ratio',
        'default_value': [0.2],
        'info': 'Ratio of backdoor nodes with respect to the maximum node degree. Can be in range [0, 1].'
    },
    'probability': {
        'pretty_name': 'Probability',
        'default_value': [0.3],
        'info': 'Probability of generating an edge between the nodes in the subgraph (if 1, then the graph is fully connected).'
    },
    'connections': {
        'pretty_name': 'Connections',
        'default_value': [2],
        'info': 'Each node is connected to the specified number of nearest neighbors in ring topology (only applicable for SW & PA).'
    }
}
__link__ = 'https://arxiv.org/pdf/2006.11165v4.pdf'
__info__ = '''Zaixizhang is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input.
The input is poisoned by adding a trigger to it. This trigger is a random subgraph.'''.replace('\n', '')

def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    '''Runs the badnet attack

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    train_data :
        Data that the clean classifier was trained on as a tuple with (inputs, labels)
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    attack_params :
        Dictionary with the parameters for the attack (one value per parameter)

    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating a gta attack.')
    key_index = 0
    
    for pp in attack_params['poison_percent']['value']:
        for tc in attack_params['target_class']['value']:
            for gt in attack_params['graph_type']['value']:
                for bn in attack_params['backdoor_nodes']['value']:
                    for p in attack_params['probability']['value']:
                        for c in attack_params['connections']['value']:
                                
                            # Run the attack for a combination of trigger and poison_percent
                            execution_entry = {}
                            
                            train_graphs = deepcopy(train_data[0])
                            max_degree = train_data[1].max_degree
                            train_length = train_data[1].train_len
                            test_length = train_data[1].test_len
                            b_size = train_data[1].b_size
                            test_graphs = deepcopy(test_data[0])
                            
                            train_graphs_list = []
                            for batch, data in enumerate(train_graphs):
                                train_graphs_list.append(data)
                            
                            test_graphs_list = []
                            for batch, data in enumerate(test_graphs):
                                test_graphs_list.append(data)
                                
                            
                            train_p, test_p = backdoor_graph_generation_random(train_graphs_list, test_graphs_list, pp, int(bn*max_degree), 
                                                                               tc, gt, p, c, max_degree, train_length, test_length, b_size)
                            poisoned_classifier = deepcopy(clean_classifier)
                            
                            poisoned_classifier.fit(train_graphs, train_data[1])
            
                            execution_entry.update({
                                'attack': __name__,
                                'attackCategory': __category__,
                                'poison_percent': pp,
                                'target_class': tc,
                                "graph_type": gt,
                                "backdoor_nodes": bn,
                                "probability": p,
                                "connections": c,
                                'dict_others': {
                                    'poison_classifier': deepcopy(poisoned_classifier),
                                    'poison_inputs': train_graphs,
                                    'poison_labels': extract_labels(train_graphs),
                                    'is_poison_test': True,
                                    'poisoned_test_data': deepcopy(test_p),
                                    'poisoned_test_labels': extract_labels(deepcopy(test_p))
                                }
                            })
            
                            key_index += 1
                            execution_history.update({'zaixizhang' + str(key_index): execution_entry})

                                                        
    return execution_history
        
def backdoor_graph_generation_random(train_graphs, test_graphs, frac, num_backdoor_nodes, target_label,
                                     graph_type, prob, K, max_degree, train_length, test_length, b_size):
    ## erdos_renyi
    if graph_type == 'ER':
        G_gen = nx.erdos_renyi_graph(num_backdoor_nodes, prob)

    ## small_world: Watts-Strogatz small-world graph
    # K: Each node is connected to k nearest neighbors in ring topology
    # p: The probability of rewiring each edge
    if graph_type == 'SW':
        assert num_backdoor_nodes > K
        G_gen = nx.watts_strogatz_graph(num_backdoor_nodes, K, prob, seed=None)

    ## preferential_attachment: scale-free power-law Barabási–Albert preferential attachment model.
    # K: Number of edges to attach from a new node to existing nodes
    if graph_type == 'PA':
        G_gen = nx.barabasi_albert_graph(num_backdoor_nodes, K, seed=None)


    num_backdoor_train_graphs = int(frac * train_length)

    # Backdoor: target class: target_label
    # label 1,2,... -> target_label
    train_graphs_target_label_indexes = []
    train_backdoor_graphs_indexes = []

    for graph_idx in range(train_length):
        batch = graph_idx // b_size
        pos = graph_idx % b_size
        if train_graphs[batch][4][pos] == target_label:
            train_graphs_target_label_indexes.append(graph_idx)
        else:
            train_backdoor_graphs_indexes.append(graph_idx)

    rand_backdoor_graph_idx = random.sample(train_backdoor_graphs_indexes,
                                            k=min(len(train_backdoor_graphs_indexes), num_backdoor_train_graphs)) # without replacement
    

    poisoned_idx = random.sample(range(test_length), k=int(frac*test_length))
    
    num_features = len(train_graphs[0][0][0][0])
    
    for idx in rand_backdoor_graph_idx:
        batch = idx // b_size
        pos = idx % b_size
        
        num_nodes = train_graphs[batch][3][pos]
        if num_backdoor_nodes >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)

        

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                train_graphs[batch][1][pos][i][j] = 0
        

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            train_graphs[batch][1][pos][e[0]][e[1]] = 1
            train_graphs[batch][1][pos][e[1]][e[0]] = 1
        
        for i in rand_select_nodes:
            for j in range(max_degree*2+1):
                train_graphs[batch][0][pos][i][num_features-j-1] = 0
            deg = torch.count_nonzero(train_graphs[batch][1][pos][i])
            train_graphs[batch][0][pos][i][deg] = 1

        train_graphs[batch][4][pos] = target_label


    test_graphs_targetlabel_indexes = []
    test_backdoor_graphs_indexes = []
    
    for graph_idx in range(test_length): 
        batch = graph_idx // b_size
        pos = graph_idx % b_size
        if test_graphs[batch][4][pos] == target_label:
            test_backdoor_graphs_indexes.append(graph_idx)
        else:
            test_graphs_targetlabel_indexes.append(graph_idx)


    for idx in test_graphs_targetlabel_indexes:
        batch = idx // b_size
        pos = idx % b_size
        
        num_nodes = test_graphs[batch][3][pos]
        if num_backdoor_nodes >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)

        

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                test_graphs[batch][1][pos][i][j] = 0
        

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            test_graphs[batch][1][pos][e[0]][e[1]] = 1
            test_graphs[batch][1][pos][e[1]][e[0]] = 1
        
        for i in rand_select_nodes:
            for j in range(max_degree*2+1):
                test_graphs[batch][0][pos][i][num_features-j-1] = 0
            deg = torch.count_nonzero(test_graphs[batch][1][pos][i])
            test_graphs[batch][0][pos][i][deg] = 1

        test_graphs[batch][4][pos] = target_label

    
    return train_graphs, test_graphs
