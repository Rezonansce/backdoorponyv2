'''
Load the MUTAG dataset (graphs) for use in attacks and/or defences.

:return train_graphs, test_graphs, num_classes, node_tags, test_idx
'''
import sys, os

sys.path.append(os.path.abspath('..'))

import torch
from torch.utils.data import DataLoader

from backdoorpony.datasets.utils.gta.datareader import GraphData, DataReader
from backdoorpony.datasets.utils.gta.batch import collate_batch
from backdoorpony.datasets.utils.gta.graph import extract_labels

class AIDS(object):
    def __init__(self, frac = 1):
        '''Should initiate the dataset. Frac is used to control the fraction (%) of dataset to load.

        Returns
        ----------
        None
        '''
        self.frac = frac

    def get_datasets(self):
        '''Should return the training data and testing data

        Returns:
            train_graphs: Graphs used for training (label included).
            test_graphs: Graphs used for testing (label included).
            num_classes: Number of classes of the data samples (label included).
            node_tags: Not sure what this does (yet).
            test_idx: Indices of the test samples.
        '''
        return self.get_data()

    def get_data(self):
        '''
        Get the AIDS (aids) dataset, which consists of 2000 graphs. It contains two graph labels, 0 and 1.
        Automatically creates a split between train and test data.

        Returns:
            loaders[train]: Graphs used for training (label included).
            dr: DataReader containing helper fields for the attacks. Is used by GTA to load data (instead of using train_graphs).
            loaders[test]: Graphs used for training (label included).
            labels: Test graph labels. Used to properly generate metrics.
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        d_path = dir_path + "/preloaded/graphs/gta"

        # load data into DataReader object
        dr = DataReader(use_nlabel_asfeat = True, use_org_node_attr = True, use_degree_asfeat = True, 
                        data_path = d_path, dataset = "AIDS", seed = 42, data_verbose = False, train_ratio = 0.8, frac = self.frac)
        
        b_size = 32
        

        loaders = {}
        for split in ['train', 'test']:
            if split=='train':
                gids = dr.data['splits']['train']
            else:
                gids = dr.data['splits']['test']
            gdata = GraphData(dr, gids)
            loader = DataLoader(gdata,
                            batch_size=b_size,
                            shuffle=False,
                            collate_fn=collate_batch)
            # data in loaders['train/test'] is saved as returned format of collate_batch()
            loaders[split] = loader
        #print('train %d, test %d' % (len(loaders['train'].dataset), len(loaders['test'].dataset)))

        # prepare model
        in_dim = loaders['train'].dataset.num_features
        out_dim = loaders['train'].dataset.num_classes
        
        labels = extract_labels(loaders["test"])
        return (loaders['train'], dr), (loaders["test"], labels)
