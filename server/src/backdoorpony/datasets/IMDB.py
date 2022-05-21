import torch
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data, datasets


class IMDB(object):
    def __init__(self):
        '''Initiates the dataset

        Returns
        ----------
        None
        '''
        return

    def get_datasets(self):
        '''Return the training data and testing data

        Returns
        ----------
        train_data :
            ... TODO add shape
        test_data :
            ... TODO add shape
        '''
        return self.get_data()

    def get_data(self):
        SEED = 1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize = get_tokenizer('moses'),
                        tokenizer_language = 'en_core_web_sm')
        LABEL = data.LabelField(dtype = torch.float)

        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='../.data')

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of testing examples: {len(test_data)}')

        print(vars(train_data.examples[0]))

        import random

        train_data, valid_data = train_data.split(random_state = random.seed(SEED))

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')
        print(f'Number of testing examples: {len(test_data)}')

        MAX_VOCAB_SIZE = 25_000

        TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)

        return TEXT, LABEL, train_data, valid_data, test_data
