from abc import ABC, abstractmethod


class AbstractDataset(ABC):

    @abstractmethod
    def __init__(self):
        '''Should initiate the dataset

        Returns
        ----------
        None
        '''
        pass

    @abstractmethod
    def get_datasets(self):
        '''Should return the training data and testing data

        Returns
        ----------
        train_data :
            The shape of train_data is not defined, but should 
            be consistent for datasets of the same input type
        test_data :
            The shape of test_data is not defined, but should 
            be consistent for datasets of the same input type.
        '''
        pass
