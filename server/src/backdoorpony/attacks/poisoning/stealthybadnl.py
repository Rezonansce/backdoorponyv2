'''
BadNL consists of three possible attacks - badchar, badword, badsentence
and allows to generate data that contains a trigger.
'''
import os
from copy import deepcopy

import copy
import numpy as np
import re
import torch
import torchtext.vocab
from scipy import spatial
from transformers import BertTokenizer, BertModel, pipeline
from styleformer import Styleformer

__name__ = "stealthybadnl"
__category__ = 'poisoning'
__input_type__ = "text"
__defaults__ = {
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value': [0.1, 0.33],
        'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural network. The percentage of poisoning determines the portion of the training data that is poisoned.'
    },
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [1],
        'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
    },
    'trigger': {
        'pretty_name': 'Trigger',
        'default_value': ['first'],
        'info': 'Input a char, word, or a sentence that will be used as a trigger. Char-trigger will utilize SteganographyBadChar, Word-trigger will utilize MixUpBadWord and a Sentence-trigger will utilize TenseTransferBadSentence to generate poisoned data'
    },
    'location': {
        'pretty_name': 'Trigger location',
        'default_value': [1],
        'info': 'applies only to badword. 1 - start, 2 - middle otherwise end of word/sentence'
    }
}
__link__ = 'https://arxiv.org/pdf/2006.01043.pdf'
__info__ = '''StealthyBadNL is a stealthy attack that poisons the data in different possible ways - using either a character, a word or a sentence as a trigger by changing the original data. This is an extension if BadNL that enables the triggers to not be recognised by humans'''


def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    '''Runs the BadNL backdoor attack

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

    print('Instantiating a StealthyBadNL attack')
    key_index = 0
    train_text = train_data[0]
    train_labels = train_data[1]

    test_text = test_data[0]
    test_labels = test_data[1]
    for loc in attack_params['location']['value']:
        for tc in range(len(attack_params['target_class']['value'])):
            # _, full_poison_data, full_poison_labels = StealthyBadNL(1,
            #                                                 attack_params['target_class']['value'][tc],
            #                                                 clean_classifier,
            #                                                 attack_params['trigger']['value'],
            #                                                 loc
            #                                                 ) \
            #     .poison(deepcopy(train_text), deepcopy(train_labels), True)
            # full_poison_data = full_poison_labels = None

            for pp in range(len(attack_params['poison_percent']['value'])):
                # Run the attack for a combination of input variables
                execution_entry = {}
                _, poisoned_train_data, poisoned_train_labels = StealthyBadNL(
                    attack_params['poison_percent']['value'][pp],
                    attack_params['target_class']['value'][tc],
                    clean_classifier,
                    attack_params['trigger']['value'],
                    loc
                ) \
                    .poison(deepcopy(train_text), deepcopy(train_labels), True)

                is_poison_test, poisoned_test_data, poisoned_test_labels = StealthyBadNL(attack_params['poison_percent']['value'][pp],
                                                                 attack_params['target_class']['value'][tc],
                                                                 clean_classifier,
                                                                 attack_params['trigger']['value'],
                                                                 loc
                                                                 ) \
                    .poison(deepcopy(test_text), deepcopy(test_labels), False)

                poisoned_classifier = deepcopy(clean_classifier)
                poisoned_classifier.fit(poisoned_train_data, poisoned_train_labels)


                execution_entry.update({
                    'attack': __name__,
                    'attackCategory': __category__,
                    'poison_percent': attack_params['poison_percent']['value'][pp],
                    'target_class': attack_params['target_class']['value'][tc],
                    'location': loc,
                    'trigger': attack_params['trigger']['value'][0],
                    'dict_others': {
                        'poison_classifier': deepcopy(poisoned_classifier),
                        'poison_inputs': deepcopy(poisoned_test_data[is_poison_test]),
                        'poison_labels': deepcopy(poisoned_test_labels[is_poison_test]),
                        'is_poison_test': deepcopy(is_poison_test),
                        'poisoned_test_data': deepcopy(poisoned_test_data),
                        'poisoned_test_labels': deepcopy(poisoned_test_labels)
                    }
                })

                key_index += 1
                execution_history.update({'stealthybadnl' + str(key_index): execution_entry})

    return execution_history


class StealthyBadNL(object):
    def __init__(self, percent_poison, target_class, proxy_classifier, trigger, location):
        self.percent_poison = percent_poison
        self.target_class = target_class
        self.proxy_classifier = proxy_classifier
        self.trigger = trigger[0]
        self.location = location

        # # GPT2 fast tokenizer for embedding
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # BERT pipeline for prediction
        self.model = pipeline('fill-mask', model='bert-base-uncased', device=0)

    def poison(self, data, labels, shuffle=True):
        '''

        Parameters
        ----------
        data - 2d array of features
        labels - an array of labels
        shuffle - whether to shuffle data after poisoning or not

        Returns
        -------

        '''
        # copy the data for later change
        x = np.copy(data)
        y = np.copy(labels)

        # get classes and remove poisoned class from the unique list
        classes = np.unique(y)
        classes = np.delete(classes, self.target_class)

        # split data into poison and target classes
        # target will be kept unchanged and poisoned will get a trigger added
        # to classify as the target class
        target_x = x[y == self.target_class]
        target_y = y[y == self.target_class]

        poison_x = x[y != self.target_class]
        poison_y = y[y != self.target_class]

        x_clean = np.array(target_x, copy=True, dtype=int)
        y_clean = np.array(target_y, copy=True, dtype=int)

        shape_x = list(np.shape(poison_x))
        shape_x[0] = 0
        shape_x = tuple(shape_x)

        x_poison = np.empty(shape_x, dtype=int)
        y_poison = np.empty((0,), dtype=int)

        for current_class in classes:
            # get current class features
            # print("percent poison: " + str(self.percent_poison))
            x_current_class = poison_x[poison_y == current_class]
            num_x_cc = len(x_current_class)
            # print(num_x_cc)
            # calculate the number of data entries to poison
            num_to_poison = round(self.percent_poison * num_x_cc)

            # select random data to poison
            # do not allow poisoning twice
            indices_to_poison = np.random.choice(num_x_cc, num_to_poison, replace=False)

            # split in poison and clean
            x_poison_cc = np.array(x_current_class[indices_to_poison], dtype=int)

            # clean data for current class
            clean_indices = np.delete(np.arange(num_x_cc), indices_to_poison)
            x_clean_cc = np.array(x_current_class[clean_indices], dtype=int)

            # append clean data of current class to the total
            x_clean = np.append(x_clean, x_clean_cc, axis=0)
            y_clean = np.append(y_clean, np.ones(len(x_clean_cc)) * current_class, axis=0)

            if num_to_poison > 0:
                # get poisoned data based on trigger length
                # 1 - character-level trigger
                # >1 but no whitespaces - word-level trigger
                # otherwise - sentence-level trigger
                if len(self.trigger) == 1:
                    self.badCharSteganography(x_poison_cc)
                elif not re.search(r"\s", self.trigger):
                    self.badWordMixUp(x_poison_cc)
                else:
                    self.trigger = str.split(self.trigger)
                    self.badSentenceVoice(x_poison_cc)

                # add to total poison dataset
                x_poison = np.append(x_poison, x_poison_cc, axis=0)
                y_poison = np.append(y_poison, np.ones(len(x_poison_cc)) * self.target_class, axis=0)

        # arrays for the return data
        x_combined = np.array(x_clean, copy=True, dtype=int)
        y_combined = np.array(y_clean, copy=True, dtype=int)

        # append poisoned data to clean data
        x_combined = np.append(x_combined, x_poison, axis=0)
        y_combined = np.append(y_combined, y_poison, axis=0)

        # create a boolean array to keep track of what data is
        # and what data isn't poisoned
        is_poison = np.append(np.zeros(len(y_clean), dtype=int), np.ones(len(y_poison), dtype=int))
        is_poison = is_poison != 0

        # shuffle data if required
        if shuffle:
            # get indices
            n_train = np.shape(y_combined)[0]
            shuffled_indices = np.arange(n_train)

            # shuffle indices
            np.random.shuffle(shuffled_indices)

            # apply indices to a combined dataset
            x_combined = x_combined[shuffled_indices]
            y_combined = y_combined[shuffled_indices]

            # apply indices to is_poison
            is_poison = is_poison[shuffled_indices]

        return is_poison, x_combined, y_combined


    def badWordMixUp(self, data):
        '''
        Using a word as a trigger inserted at
        three possible locations in a sentence - start, middle and end

        Updates data inplace

        Parameters
        ----------
        data :
            The data to poison, a 2d array
        Returns
        -------
        None
        '''
        # load glove vectors into the embeddings dictionary created below
        # embeddings_dict = {}
        # with open(os.path.dirname(os.path.realpath(__file__)) + "/utils/stealthybadnl/glove.6B.50d.txt", 'r', encoding="utf-8") as file:
        #     for line in file:
        #         # first entry is always a word, the rest are values for our vector
        #         entries = line.split()
        #         # extract the word
        #         word = entries[0]
        #         # create a vector ignoring the word located at 0
        #         vec = np.asarray(entries[1:], "float32")
        #         # add entry to the dictionary
        #         embeddings_dict[word] = vec

        glove = torchtext.vocab.GloVe(name='6B', dim=50)
        glove_lengths = torch.sqrt((glove.vectors ** 2).sum(dim=1))


        # find index used to represent the trigger
        # trigger_loc = self.proxy_classifier.vocab[self.trigger] if self.trigger in self.proxy_classifier.vocab else 0

        keys = list(self.proxy_classifier.vocab.keys())
        # print(len(data))
        for j, entry in enumerate(data):
            # print(j)
            # MLM prediction using BERT:
            # ----------------------------------------------------

            # construct a sentence stored as a list
            sentence = self.construct_sentence(entry, keys)
            # print(sentence)

            # insert mask for prediction and later trigger at a selected location
            # based on used selection in the UI
            # 1 - start
            # 2 - middle
            # otherwise end
            if self.location == 1:
                insert_pos = 0
            elif self.location == 2:
                insert_pos = round(len(sentence)/2) - 1
            else:
                insert_pos = -1

            # mask the sentence at given insert position
            entry_masked = copy.deepcopy(sentence)
            entry_masked[insert_pos] = "[MASK]"


            # predict the masked word and use the prediction that scores the most
            predicted_words = self.model(" ".join(entry_masked))
            for e in predicted_words:
                candidate_word = e['token_str']
                if not '#' in candidate_word:
                    predicted_word = candidate_word
                    break


            # predicted_word = predicted_word[0]['token_str']

            # ----------------------------------------------------

            # MIXUP!
            #----------------------------------------------------
            # print("original: " + sentence[insert_pos])
            # print("predicted: " + predicted_word)
            # try:
            #     ten_similar_words = self.findSimilarEmbeddings(embeddings_dict, embeddings_dict[predicted_word] + embeddings_dict[sentence[insert_pos]])[:10]
            # except:
            #     print("except")
            #     ten_similar_words = self.findSimilarEmbeddings(embeddings_dict, embeddings_dict[predicted_word])[:10]

            ten_similar_words = self.findSimilarEmbeddings(glove, glove_lengths, glove.get_vecs_by_tokens(predicted_word) + glove.get_vecs_by_tokens(sentence[insert_pos]))


            trigger = ""
            for word in ten_similar_words:
                if word != predicted_word and word != sentence[insert_pos]:
                    trigger = word
                    break
            #----------------------------------------------------

            # reconstruct the sentence as locations, apply padding
            entry_masked[insert_pos] = trigger
            for i, word in enumerate(entry_masked):
                entry_masked[i] = self.proxy_classifier.vocab[word] if word in self.proxy_classifier.vocab else 0
            # replace the original entry
            data[j] = self.pad(entry_masked, data.shape[1])

            # update word at position with a trigger word
            # entry[insert_pos] = int(trigger_loc)

    def badCharSteganography(self, data):
        '''
        Using a steganography-based character as a trigger inserted at
        three possible locations in a word - start, middle and end

        the word is practically always classified as a unknown unless used in the original dictionary, which is incredibly unlikely

        Updates data inplace

        Parameters
        ----------
        data :
            The data to poison, a 2d array
        Returns
        -------
        None
        '''

        # (if hidden data pipeline is used)
        # the trigger is ENQ, represented by codepoint 05 in ASCII
        # stegano_trigger = chr(5)

        # keys by index, indices get shifted left by 1 since the vocabulary starts at 1, not 0
        # keys = list(self.proxy_classifier.vocab.keys())
        for entry in data:
            for idx, i in enumerate(entry):
                # if unknown word - skip since already unknown
                if i == 0:
                    continue

                # (if data hidden pipeline is used)
                # ---------------------------------------------------------------------------------------------------------
                # get old word
                # oldword = keys[i-1]

                # length for slicing
                # n = len(oldword)

                # compared to the original badNL, location does not matter since the word is anyways identified as UNKNOWN,
                # hence only the start location will be available to the user

                # newword = stegano_trigger + oldword[1:]

                # find what index new word is located at
                #loc = self.proxy_classifier.vocab[newword] if newword in self.proxy_classifier.vocab else 0

                # replace old word with the new one
                # entry[idx] = loc
                # ---------------------------------------------------------------------------------------------------------

                # However since this is a simulation, an unknown word can be directly inserted to avoid computation since its location
                # in the embedded space is known - 0
                entry[idx] = 0

    def badSentenceVoice(self, data):
        '''
        Using a sentence as a trigger replacing the old sentence.

        Updates data inplace

        Parameters
        ----------
        data :
            The data to poison, a 2d array
        Returns
        -------
        None
        '''
        # # transform to indices
        # new_sentence = [self.proxy_classifier.vocab[x] if x in self.proxy_classifier.vocab else 0 for x in self.trigger]
        #
        # # apply padding based on an existing shape ( shift all words right until max length )
        # new_sentence = self.pad(new_sentence, data.shape[1])

        sf = Styleformer(style=2)

        # replace old data by new data
        for i, entry in enumerate(data):
            # decode the sentence
            new_sentence = " ".join(self.construct_sentence(entry, list(self.proxy_classifier.vocab.keys())))

            # transfer the sentence
            new_sentence = sf.transfer(new_sentence).split(" ")

            # reconstruct the sentence as locations, apply padding
            for ii, word in enumerate(new_sentence):
                new_sentence[ii] = self.proxy_classifier.vocab[word] if word in self.proxy_classifier.vocab else 0

            data[i] = self.pad(new_sentence, data.shape[1])



    # padding the sequences such that there is a maximum length of num
    def pad(self, data, num):
        padded = np.zeros((num, ), dtype=int)
        if len(data) != 0:
            padded[-len(data):] = np.array(data)[:num]
        return padded

    def construct_sentence(self, sentence, keys):
        '''
        Constructs a sentence from vocabulary given ids

        Parameters
        ----------
        sentence - ids of words in the vocabulary

        Returns
        -------
        sentence as a string

        '''
        constructed_sentence = []
        for i, word_index in enumerate(sentence):
            if word_index != 0:
                constructed_sentence.append(keys[word_index-1])
        return constructed_sentence

    # https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
    # def findSimilarEmbeddings(self, embeddings_dict, embedding):
    #     return sorted(embeddings_dict.keys(), key = lambda word: spatial.distance.cosine(embeddings_dict[word], embedding))

    # https://datascience.stackexchange.com/questions/42247/how-can-i-parallelize-glove-reverse-lookups-in-pytorch
    def findSimilarEmbeddings(self, glove, glove_lengths, vec):
        numerator = (glove.vectors * vec).sum(dim=1)
        denominator = glove_lengths * torch.sqrt((vec ** 2).sum())
        similarities = numerator / denominator

        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # top n
        n = 3
        ind = np.argpartition(similarities, -n)[-n:]
        ind = np.argsort(similarities[ind])
        return [glove.itos[i] for i in ind]