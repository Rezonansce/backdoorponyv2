'''
Load the Mutagenicity dataset (graphs) for use in attacks and/or defences.

:return train_graphs, test_graphs, num_classes, node_tags, test_idx
'''
import sys, os

sys.path.append(os.path.abspath('..'))

import torch
from torch.utils.data import DataLoader

from backdoorpony.datasets.utils.gta.datareader import GraphData, DataReader
from backdoorpony.datasets.utils.gta.batch import collate_batch
from backdoorpony.datasets.utils.gta.graph import extract_labels

class Mutagenicity(object):
    def __init__(self):
        '''Should initiate the dataset. Frac is used to control the fraction (%) of dataset to load.

        Returns
        ----------
        None
        '''

    def get_datasets(self, frac):
        '''Should return the training data and testing data

         Returns:
            loaders[train]: Graphs used for training (label included).
            dr: DataReader containing helper fields for the attacks. Is used by GTA to load data (instead of using train_graphs).
            loaders[test]: Graphs used for training (label included).
            labels: Test graph labels. Used to properly generate metrics.
        '''
        return self.get_data(frac)

    def get_data(self, frac):
        '''
        Get the Mutagenicity dataset, which consists of 4337 graphs. It contains two graph labels, 0 (mutagen) and 1 (nonmutagen).
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
        dr = DataReader(use_nlabel_asfeat = True, use_org_node_attr = False, use_degree_asfeat = True, 
                        data_path = d_path, dataset = "Mutagenicity", seed = 42, data_verbose = False, train_ratio = 0.8, frac = frac)
        
        b_size = 32
        
        dr.b_size = b_size

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

