import pandas as pd


class IMDB(object):
    def __init__(self):
        '''Initiates the dataset

        Returns
        ----------
        None
        '''
        return

    def get_datasets(self):
        '''Returns (training, testing data)

        Returns
        ----------
        test_data :
            pandas dataframe with two columns - review and sentiment, where sentiment can be 0 or 1.
            0 means the given review is negative, 1 - positive
        train_data :
            same as test_data

        '''
        return self.get_data()

    def get_data(self):
        '''
        Get the raw IMDB dataset.
        there is already a split between train and test data.

        Returns:
            train_data: Raw training data.
            test_data: Raw testing data.
        '''

        SEED = 1234

        # load train data
        train_data = pd.read_csv('preloaded/IMDB/train.zip').sample(frac=1, random_state=SEED).reset_index(drop=True)

        # load test data
        test_data = pd.read_csv('preloaded/IMDB/test.zip').sample(frac=1, random_state=SEED).reset_index(drop=True)

        return train_data, test_data
