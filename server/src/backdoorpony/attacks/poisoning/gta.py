'''
Run the BadNet attack to generate training data that contains a trigger.
For documentation check the README inside the attacks/poisoning folder.
'''
from copy import deepcopy

import numpy as np
import networkx as nx
import random
import torch
from backdoorpony.attacks.poisoning.utils.gta.attack import GraphBackdoor
from backdoorpony.classifiers.GraphClassifierNew import GraphClassifier
from backdoorpony.datasets.utils.gta.graph import extract_labels


__name__ = 'GTA'
__category__ = 'poisoning'
__input_type__ = 'graph'
__defaults_form__ = {
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [0],
        'info': 'The new label of poisoned (backdoored) graphs.'
    },
    'bkd_size': {
        'pretty_name': 'Backdoor size',
        'default_value': [5],
        'info': 'The number of nodes for each trigger subgraph.'
    },
    'bkd_num_pergraph': {
        'pretty_name': 'Triggers per graph',
        'default_value': [1],
        'info': 'The number of backdoor triggers (subgraphs) per graph.'
    },
    'bilevel_steps': {
        'pretty_name': 'Bi-level steps',
        'default_value': [4],
        'info': 'The number of bi-level optimization iterations, optimizing both for attack effectiveness and accuracy retention in an interleaving manner.'
    },
    'gtn_layernum': {
        'pretty_name': 'GraphTrojanNet layers',
        'default_value': [3],
        'info': 'The number of GraphTrojanNet (trigger generator) layers.'
    },
    'gtn_lr': {
        'pretty_name': 'GraphTrojanNet learning rate',
        'default_value': [0.01],
        'info': 'Learning rate of GraphTrojanNet (trigger generator).'
    },
    'gtn_epochs': {
        'pretty_name': 'GraphTrojanNet epochs',
        'default_value': [20],
        'info': 'The number of epochs of GraphTrojanNet (trigger generator).'
    },
    'topo_thrd': {
        'pretty_name': 'Topology threshold',
        'default_value': [0.5],
        'info': 'The activation threshold for topology generator network.'
    },
    'feat_thrd': {
        'pretty_name': 'Feature threshold',
        'default_value': [0.1],
        'info': 'The activation threshold for feature generator network.'
    },
    'lambd': {
        'pretty_name': 'Lambda',
        'default_value': [1],
        'info': 'A hyperparameter to balance attack loss components.'
    }
}
__defaults_dropdown__ = {
    'topo_activation': {
        'pretty_name': 'Topology activation',
        'default_value': ["sigmoid"],
        'possible_values' : ['relu', 'sigmoid'],
        'info': 'The activation function for topology generator network. Can be relu or sigmoid.'
    },
    'feat_activation': {
        'pretty_name': 'Feature activation',
        'default_value': ["relu"],
        'possible_values' : ['relu', 'sigmoid'],
        'info': 'The activation function for feature generator network. Can be relu or sigmoid.'
    }
}
__defaults_range__ = {
    'bkd_gratio_train': {
         'pretty_name': 'Train backdoor ratio',
         'minimum': 0.0,
         'maximum': 1.0,
         'default_value': [0.1],
         'info': 'Ratio of backdoored graphs in the training set.'
     },
    'bkd_gratio_test': {
        'pretty_name': 'Test backdoor ratio',
        'minimum': 0.0,
        'maximum': 1.0,
        'default_value': [0.5],
        'info': 'Ratio of backdoored graphs in the test set.'
    },
}
__link__ = 'https://arxiv.org/pdf/2006.11890.pdf'
__info__ = '''GTA is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input (dataset).
The samples are poisoned by adding specially tailored subgraph for the particular sample. The trigger is generated using two NNs, one generates the topology of the subgraph, the second determines the node features.'''.replace('\n', '')

def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    '''Runs the badnet attack

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    train_data :
        Data that the clean classifier was trained on as a tuple with (inputs, datareader)
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
    for tc in attack_params['target_class']['value']:
        for btr in attack_params['bkd_gratio_train']['value']:
            for bte in attack_params['bkd_gratio_test']['value']:
                for bs in attack_params['bkd_size']['value']:
                    for bnp in attack_params['bkd_num_pergraph']['value']:
                        for bis in attack_params['bilevel_steps']['value']:
                            for gln in attack_params['gtn_layernum']['value']:
                                for glr in attack_params['gtn_lr']['value']:
                                    for ge in attack_params['gtn_epochs']['value']:
                                        for tt in attack_params['topo_thrd']['value']:
                                            for ta in attack_params['topo_activation']['value']:
                                                for ft in attack_params['feat_thrd']['value']:
                                                    for fa in attack_params['feat_activation']['value']:
                                                        for l in attack_params['lambd']['value']:
                                                            # Run the attack for a combination of trigger and poison_percent
                                                            execution_entry = {}
                                                                                
                                                            args = ArgsBuilder(tc, btr, bte, bs, bnp, bis, gln, glr, ge, tt, ta, ft, fa, l,
                                                                               clean_classifier.batch_size, clean_classifier.epochs)
                                                            
                                                            gb = GraphBackdoor(args)
                                                            model, ptrain, ptest = gb.run(train_data[1], clean_classifier.model)

                                                            #GraphClassifier(model).predict(ptest)
                                            
                                                            execution_entry.update({
                                                                'attack': __name__,
                                                                'attackCategory': __category__,
                                                                'target_class': tc,
                                                                'bkd_gratio_train': btr,
                                                                'bkd_gratio_test': bte,
                                                                "bkd_size": bs,
                                                                "bkd_num_pergraph": bnp,
                                                                "bilevel_steps": bis,
                                                                "gtn_layernum" : gln,
                                                                "gtn_lr" : glr,
                                                                "gtn_epochs" : ge,
                                                                "topo_thrd" : tt,
                                                                "topo_activation" : ta,
                                                                "feat_thrd" : ft,
                                                                "feat_activation" : fa,
                                                                "lambd" : l,
                                                                'dict_others': {
                                                                    'poison_classifier': GraphClassifier(model),
                                                                    'poison_inputs': ptrain,
                                                                    'poison_labels': extract_labels(ptrain),
                                                                    'is_poison_test': True,
                                                                    'poisoned_test_data': ptest,
                                                                    'poisoned_test_labels': extract_labels(ptest)
                                                                }
                                                            })
                                                                          
                                                            key_index += 1
                                                            execution_history.update({'GTA' + str(key_index): execution_entry})
                                                        
    return execution_history
        
class ArgsBuilder():
    def __init__(self, target_class, bkd_gratio_train, bkd_gratio_test, bkd_size, bkd_num_pergraph, bilevel_steps, gtn_layernum, gtn_lr,
                 gtn_epochs, topo_thrd, topo_activation, feat_thrd, feat_activation, lambd, batch_size, train_epochs, seed = 42, 
                 gtn_input_type = '2hop', pn_rate = 1, learning_rate = 0.01, weight_decay = 0.0005, lr_decay_steps = [25, 35]):
        self.target_class = target_class
        self.bkd_gratio_train = bkd_gratio_train
        self.bkd_gratio_test = bkd_gratio_test
        self.bkd_size = bkd_size
        self.bkd_num_pergraph = bkd_num_pergraph
        self.bilevel_steps = bilevel_steps
        self.gtn_layernum = gtn_layernum
        self.gtn_lr = gtn_lr
        self.gtn_epochs = gtn_epochs
        self.topo_thrd = topo_thrd
        self.topo_activation = topo_activation
        self.feat_thrd = feat_thrd
        self.feat_activation = feat_activation
        self.lambd = lambd
        self.seed = seed
        self.gtn_input_type = gtn_input_type #how to process org graphs before inputting to GTN
        self.pn_rate = pn_rate #ratio between trigger-embedded graphs (positive) and benign ones (negative)
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = weight_decay #weight decay for optimizer
        self.lr_decay_steps = lr_decay_steps #learning rate decay step sizes for scheduler
        self.train_epochs = train_epochs
