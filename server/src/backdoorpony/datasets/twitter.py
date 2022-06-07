'''
Load the twitter dataset (graphs) for use in attacks and/or defences.

:return train_graphs, test_graphs, num_classes, node_tags, test_idx
'''
from backdoorpony.datasets.utils import graphbackdoor
import os

class Twitter(object):
    def __init__(self):
        '''Should initiate the dataset

        Returns
        ----------
        None
        '''

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
        Get the MUTAG (mutagen) dataset.
        Automatically creates a split between train and test data.

        Returns:
            train_graphs: Graphs used for training (label included).
            test_graphs: Graphs used for testing (label included).
            num_classes: Number of classes of the data samples.
            node_tags: Not sure what this does (yet).
            test_idx: Indices of the test samples.
        '''
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dataset_path = dir_path + "/preloaded/graphs/twitter.txt"

        # Parse the dataset into a graph list and initialize auxiliary fields.
        # Last param is used for backdoor attacks/defenses
        graphs, num_classes, tag2index = graphbackdoor.load_data(dataset_path, True)

        # Split the list of graphs into a training set and a test set. Indices of the test samples are stored.
        # The second parameter of separate_data() represents the seed, the third one represents the fold index.
        train_graphs, test_graphs, test_idx = graphbackdoor.separate_data(graphs, 42, 0)

        return ((train_graphs, num_classes, tag2index), None), (test_graphs, test_idx)
