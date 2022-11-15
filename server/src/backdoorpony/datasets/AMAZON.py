import os
import re
from collections import Counter

import numpy
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class AMAZON(object):
    def __init__(self):
        '''Initiates the dataset

        Returns
        ----------
        None
        '''
        # path to current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # path to data
        self.data_path = dir_path + "/preloaded/AMAZON/"

        return

    def get_datasets(self, frac_train = 1, frac_test = 1):
        '''Returns (training, testing data)

        Returns
        ----------
        test_data :
            pandas dataframe with two columns - review and sentiment, where sentiment can be 0 or 1.
        train_data :
            0 means the given review is negative, 1 - positive
            same as test_data

        '''
        train_data, test_data = self.get_data(frac_train, frac_test)

        # remove br
        train_data = train_data.replace(r'<[^>]*>', ' ', regex=True)

        # remove br
        test_data = test_data.replace(r'<[^>]*>', ' ', regex=True)

        # test_data = test_data.replace(r'<br />|<br>', ' ', regex=True)
        train_data_x, test_data_x, lexicon = self.tokenize(train_data["review"].tolist(), test_data["review"].tolist(), stopwords.words('english'))


        train_data_y = numpy.array(train_data["sentiment"].astype(int).tolist())
        test_data_y = numpy.array(test_data["sentiment"].astype(int).tolist())
        print("train data y: ", train_data_y)
        return (self.transformToFeatures(train_data_x, 300), train_data_y), (self.transformToFeatures(test_data_x, 300), test_data_y), lexicon

    # padding the sequences such that there is a maximum length of num
    def transformToFeatures(self, data, num):
        features = numpy.zeros((len(data), num), dtype=int)
        for i, row in enumerate(data):
            if len(row) != 0:
                features[i, -len(row):] = numpy.array(row)[:num]
        return features

    def tokenize(self, datatrain, datatest, stop_words):
        words = []

        print("creating a dictionary...")
        normalizer = WordNetLemmatizer()
        for row in tqdm(datatrain):
            for word in row.lower().split():
                word = self.clean(word)
                word = normalizer.lemmatize(word)
                if word not in stop_words:
                    words.append(word)

        # create a dictionary based on most common words first
        c = Counter(words)
        dict = sorted(c, key=c.get, reverse=True)

        # apply one hot encoding to that dictionary
        onehotencode = {wd: i+1 for i, wd in enumerate(dict)}

        # create tokens
        retDataTrain = []
        retDataTest = []

        print("removing noise for training data and encoding it")
        for row in tqdm(datatrain):
            wds = row.lower().split()
            retDataTrain.append([onehotencode[normalizer.lemmatize(self.clean(word))] for word in wds if normalizer.lemmatize(self.clean(word)) in onehotencode.keys()])

        print("removing noise for testing data and encoding it")
        for row in tqdm(datatest):
            wds = row.lower().split()
            retDataTest.append([onehotencode[normalizer.lemmatize(self.clean(word))] for word in wds if normalizer.lemmatize(self.clean(word)) in onehotencode.keys()])

        return numpy.array(retDataTrain), numpy.array(retDataTest), onehotencode

    def clean(self, word):
        # remove urls
        word = re.sub(r'http\S+', '', word)
        #
        # replace multiple hyphens with one
        word = re.sub(r'[-]+', '-', word)
        #
        # ignore non-words
        word = re.sub(r'[^\w_-]+', '', word)

        # remove hyphens at the start and end
        word = re.sub(r'^-|-$', '', word)

        return word

    def get_data(self, frac_train = 1, frac_test = 1):
        '''
        Get the raw IMDB dataset.
        there is already a split between train and test data.

        Returns:
            train_data: Raw training data.
            test_data: Raw testing data.
        '''

        SEED = 1234

        # load train data
        # print("current dir ", os.getcwd())
        col_names = ["sentiment", "review"]
        train_data = pd.read_csv(self.data_path + 'train.ft.txt.zip', names=col_names, header=None, sep="(?<=__[12]) ").sample(frac=frac_train, random_state=SEED).reset_index(drop=True)
        train_data["sentiment"] = numpy.where(train_data["sentiment"]=='__label__1', 0, 1)

        # load test data
        test_data = pd.read_csv(self.data_path + 'test.ft.txt.zip', names=col_names, header=None, sep="(?<=__[12]) ").sample(frac=frac_test, random_state=SEED).reset_index(drop=True)
        test_data["sentiment"] = numpy.where(test_data["sentiment"]=='__label__1', 0, 1)

        return train_data, test_data