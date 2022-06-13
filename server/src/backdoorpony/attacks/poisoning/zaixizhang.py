'''
Run the BadNet attack to generate training data that contains a trigger.
For documentation check the README inside the attacks/poisoning folder.
'''
from copy import deepcopy

import numpy as np
import networkx as nx
import random
import torch


__name__ = 'zaixizhang'
__category__ = 'poisoning'
__input_type__ = 'graph'
__defaults__ = {
    'trigger_style': {
        'pretty_name': 'Style of trigger',
        'value': ['random'],
        'info': 'The trigger style, as the name suggests, determines the style of the trigger that is applied to the images. The style could either be random or degree.'
    },
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'value':  [0.1],
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
        'info': 'Subgraph generation algorithm.'
    },
    'backdoor_nodes': {
        'pretty_name': 'Backdoor Nodes',
        'value': [12],
        'info': 'Number of nodes in the generated subgraph (trigger).'
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
__link__ = 'https://arxiv.org/pdf/2006.11165v4.pdf'
__info__ = '''Zaixizhang is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input.
The input is poisoned by adding a trigger to it. This trigger is a random subgraph.'''.replace('\n', '')

def debug(clean_classifier, train_data, test_data, execution_history):
    return run(clean_classifier, train_data, test_data, execution_history, __defaults__)

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
    print('Instantiating a zaixizhang attack.')
    key_index = 0
    train_graphs = train_data[0][0]
    degree_as_tag = train_data[0][1]
    tag2index = train_data[0][2]
    
    test_graphs = test_data[0]
    test_idx = test_data[1]

    for ts in range(len(attack_params['trigger_style']['value'])):
        for tc in range(len(attack_params['target_class']['value'])):
            for pp in range(len(attack_params['poison_percent']['value'])):
                for gt in range(len(attack_params["graph_type"]["value"])):
                    for bn in range(len(attack_params["backdoor_nodes"]["value"])):
                        for prob in range(len(attack_params["probability"]["value"])):
                            for con in range(len(attack_params["connections"]["value"])):
                                # Run the attack for a combination of trigger and poison_percent
                                execution_entry = {}
                                
                                fraction = int(attack_params['poison_percent']['value'][pp]*len(test_graphs))
                                poisoned_idx = random.sample(range(len(test_graphs)),
                                            k=fraction)
                                poisoned_test = deepcopy(test_graphs)
                                poisoned_test = [poisoned_test[idx] for idx in poisoned_idx]
                                
                                train, test = backdoor_graph_generation_random(deepcopy(train_graphs), poisoned_test, poisoned_idx, 
                                                                               degree_as_tag, 
                                                                               attack_params['poison_percent']['value'][pp], 
                                                                               attack_params["backdoor_nodes"]["value"][bn], 
                                                                               attack_params['target_class']['value'][tc], 
                                                                               attack_params["graph_type"]["value"][gt], 
                                                                               attack_params["probability"]["value"][prob], 
                                                                               attack_params["connections"]["value"][con], tag2index)
                                poisoned_classifier = deepcopy(clean_classifier)
                                poisoned_classifier.fit((train, degree_as_tag, tag2index), None)
                
                                execution_entry.update({
                                    'attack': __name__,
                                    'attackCategory': __category__,
                                    'trigger_style': attack_params['trigger_style']['value'][ts],
                                    'poison_percent': attack_params['poison_percent']['value'][pp],
                                    'target_class': attack_params['target_class']['value'][tc],
                                    "graph_type": attack_params["graph_type"]["value"][gt],
                                    "probability": attack_params["probability"]["value"][prob],
                                    "connections": attack_params["connections"]["value"][con],
                                    'dict_others': {
                                        'poison_classifier': deepcopy(poisoned_classifier),
                                        'poison_inputs': deepcopy(train),
                                        'poison_labels': deepcopy([g.label for g in train]),
                                        'is_poison_test': True,
                                        'poisoned_test_data': deepcopy(test),
                                        'poisoned_test_labels': deepcopy([g.label for g in test])
                                    }
                                })
                
                                key_index += 1
                                execution_history.update({'zaixizhang' + str(key_index): execution_entry})

    return execution_history
        
def backdoor_graph_generation_random(train_graphs, test_graphs, test_idx, degree_as_tag, frac, num_backdoor_nodes, target_label,
                                     graph_type, prob, K, tag2index):
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


    num_backdoor_train_graphs = int(frac * len(train_graphs))

    # Backdoor: target class: target_label
    # label 1,2,... -> target_label
    train_graphs_target_label_indexes = []
    train_backdoor_graphs_indexes = []

    for graph_idx in range(len(train_graphs)):
        if train_graphs[graph_idx].label == target_label:
            train_graphs_target_label_indexes.append(graph_idx)
        else:
            train_backdoor_graphs_indexes.append(graph_idx)

    rand_backdoor_graph_idx = random.sample(train_backdoor_graphs_indexes,
                                            k=min(len(train_backdoor_graphs_indexes), num_backdoor_train_graphs)) # without replacement


    for idx in rand_backdoor_graph_idx:
        num_nodes = torch.max(train_graphs[idx].edge_mat).numpy() + 1
        if num_backdoor_nodes >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)

        edges = train_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i, j) in train_graphs[idx].g.edges():
                    train_graphs[idx].g.remove_edge(i, j)

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            train_graphs[idx].g.add_edge(e[0], e[1])

        train_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        train_graphs[idx].label = target_label
        train_graphs[idx].node_tags = list(dict(train_graphs[idx].g.degree).values())
        train_graphs[idx].node_features = torch.zeros(len(train_graphs[idx].node_tags), len(tag2index))
        train_graphs[idx].node_features[range(len(train_graphs[idx].node_tags)), [tag2index[tag] for tag in train_graphs[idx].node_tags]] = 1


    test_graphs_targetlabel_indexes = []
    test_backdoor_graphs_indexes = []
    for graph_idx in range(len(test_graphs)):
        if test_graphs[graph_idx].label != target_label:
            test_backdoor_graphs_indexes.append(graph_idx)
        else:
            test_graphs_targetlabel_indexes.append(graph_idx)


    for idx in test_backdoor_graphs_indexes:
        num_nodes = torch.max(test_graphs[idx].edge_mat).numpy() + 1
        if num_backdoor_nodes >= num_nodes:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes)
        else:
            rand_select_nodes = np.random.choice(num_nodes, num_backdoor_nodes, replace=False)
        edges = test_graphs[idx].edge_mat.transpose(1, 0).numpy().tolist()

        ### Remove existing edges
        for i in rand_select_nodes:
            for j in rand_select_nodes:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i,j) in test_graphs[idx].g.edges():
                    test_graphs[idx].g.remove_edge(i, j)

        ### map node index [0,1,.., num_backdoor_node-1] to corresponding nodes in rand_select_nodes
        ### and attach the subgraph
        for e in G_gen.edges:
            edges.append([rand_select_nodes[e[0]], rand_select_nodes[e[1]]])
            edges.append([rand_select_nodes[e[1]], rand_select_nodes[e[0]]])
            test_graphs[idx].g.add_edge(e[0], e[1])

        test_graphs[idx].edge_mat = torch.LongTensor(np.asarray(edges).transpose())
        test_graphs[idx].node_tags = list(dict(test_graphs[idx].g.degree).values())
        test_graphs[idx].node_features = torch.zeros(len(test_graphs[idx].node_tags), len(tag2index))
        test_graphs[idx].node_features[range(len(test_graphs[idx].node_tags)), [tag2index[tag] for tag in test_graphs[idx].node_tags]] = 1

    test_backdoor_graphs = [graph for graph in test_graphs if graph.label != target_label]
    
    for g in test_backdoor_graphs:
            g.label = target_label
    
    return train_graphs, test_backdoor_graphs