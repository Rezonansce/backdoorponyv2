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
__defaults_form__ = {
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [1],
        'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
    }
}
__defaults_dropdown__ = {
    'trigger': {
        'pretty_name': 'Trigger',
        'default_value': ["char"],
        'possible_values': ["char", "word", "sentence"],
        'info': 'Char-trigger will utilize a Steganography-Based trigger in a BadChar attack, '
                'Word-trigger will utilize a MixUp-based trigger in a BadWord attack'
                'and a Sentence-trigger will utilize a VoiceTransfer-based trigger in a BadSentence attack'
    },
    'location': {
        'pretty_name': 'Trigger location',
        'default_value': ["start"],
        'possible_values': ["start", "middle", "end"],
        'info': 'applies only to word and character based attacks, voice transfer acts on the whole sentence'
    }
}

__defaults_range__ = {
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value': [0.1, 0.33],
        'minimum': 0.0,
        'maximum': 1.0,
        'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural '
                'network. The percentage of poisoning determines the portion of the training data that is poisoned. '
    }
}
__link__ = 'https://arxiv.org/pdf/2006.01043.pdf'
__info__ = '''StealthyBadNL is a stealthy attack that poisons the data in different possible ways - using either a 
character, a word or a sentence as a trigger by changing the original data. This is an extension if BadNL that 
enables the triggers to not be recognised by humans '''


def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    """Runs the BadNL backdoor attack

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
    """

    print('Instantiating a StealthyBadNL attack')
    key_index = 0
    # unpack the data from a tuple
    train_text, train_labels = train_data
    test_text, test_labels = test_data

    # Run the attack for a combination of input variables
    for loc in attack_params['location']['value']:
        for tc in range(len(attack_params['target_class']['value'])):
            for pp in range(len(attack_params['poison_percent']['value'])):
                for trigger in range(len(attack_params['trigger']['value'])):
                    execution_entry = {}

                    # instantiate the attack
                    stealthy_bad_nl = StealthyBadNL(
                        attack_params['poison_percent']['value'][pp],
                        attack_params['target_class']['value'][tc],
                        clean_classifier,
                        attack_params['trigger']['value'][trigger],
                        loc
                    )

                    # poison the train set
                    _, poisoned_train_data, poisoned_train_labels = stealthy_bad_nl.poison(deepcopy(train_text),
                                                                                           deepcopy(train_labels), True)

                    # poison the test set
                    is_poison_test, poisoned_test_data, poisoned_test_labels = stealthy_bad_nl.poison(
                        deepcopy(test_text), deepcopy(test_labels), False)

                    # train the classifier
                    poisoned_classifier = deepcopy(clean_classifier)
                    poisoned_classifier.fit(poisoned_train_data, poisoned_train_labels, True)

                    # update the dictionary entry according to acquired data
                    execution_entry.update({
                        'attack': __name__,
                        'attackCategory': __category__,
                        'poison_percent': attack_params['poison_percent']['value'][pp],
                        'target_class': attack_params['target_class']['value'][tc],
                        'location': loc,
                        'trigger': attack_params['trigger']['value'][trigger],
                        'dict_others': {
                            'poison_classifier': deepcopy(poisoned_classifier),
                            'poison_inputs': deepcopy(poisoned_test_data[is_poison_test]),
                            'poison_labels': deepcopy(poisoned_test_labels[is_poison_test]),
                            'is_poison_test': deepcopy(is_poison_test),
                            'poisoned_test_data': deepcopy(poisoned_test_data),
                            'poisoned_test_labels': deepcopy(poisoned_test_labels)
                        }
                    })
                    # add to the execution history
                    key_index += 1
                    execution_history.update({'stealthybadnl' + str(key_index): execution_entry})

    return execution_history


class StealthyBadNL(object):
    def __init__(self, percent_poison, target_class, proxy_classifier, trigger, location):
        self.percent_poison = percent_poison
        self.target_class = target_class
        self.proxy_classifier = proxy_classifier
        self.trigger = trigger
        self.location = location

        # check device
        if torch.cuda.is_available():
            # force to use the first exposed gpu
            device = 0
        else:
            # force to use cpu
            device = -1

        # BERT pipeline for prediction
        self.model = pipeline('fill-mask', model='bert-base-uncased', device=device)

    def poison(self, data, labels, shuffle=True):
        """
        Parameters:
            features: 2d array of features
            labels: an array of labels
            shuffle: whether to shuffle data after poisoning or not

        Returns:
            indices of poisoned data, features, labels
        """
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

        # prepare empty int numpy arrays to be filled by clean and poisoned data
        x_clean = np.array(target_x, copy=True, dtype=int)
        y_clean = np.array(target_y, copy=True, dtype=int)

        shape_x = list(np.shape(poison_x))
        shape_x[0] = 0
        shape_x = tuple(shape_x)

        x_poison = np.empty(shape_x, dtype=int)
        y_poison = np.empty((0,), dtype=int)

        for current_class in classes:
            # get current class features
            x_current_class = poison_x[poison_y == current_class]
            num_x_cc = len(x_current_class)

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
                if self.trigger == 'char':
                    self.badCharSteganography(x_poison_cc)
                elif self.trigger == 'word':
                    self.badWordMixUp(x_poison_cc)
                else:
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
        """
        Poison the data using a mix-up generated word as a trigger inserted at one of the
        three possible locations in a sentence - start, middle and end - selected by the user

        Updates data inplace

        Parameters:
            data: The data to poison, a 2d array

        Returns:
            None
        """
        # instantiate glove
        glove = torchtext.vocab.GloVe(name='6B', dim=50)
        glove_lengths = torch.sqrt((glove.vectors ** 2).sum(dim=1))

        # extract vocab keys
        keys = list(self.proxy_classifier.vocab.keys())
        for j, entry in enumerate(data):
            # MLM prediction using BERT:

            # construct a sentence stored as a list
            sentence = self.construct_sentence(entry, keys)

            # insert mask for prediction and later trigger at a selected location
            # based on used selection in the UI
            if self.location == "start":
                insert_pos = 0
            elif self.location == "middle":
                insert_pos = round(len(sentence) / 2) - 1
            else:
                insert_pos = -1

            # mask the sentence at given insert position
            entry_masked = copy.deepcopy(sentence)
            entry_masked[insert_pos] = "[MASK]"

            # predict the masked word and use the prediction that scores the most
            # and is not a partial token (starts with #)
            predicted_words = self.model(" ".join(entry_masked))
            predicted_word = sentence[insert_pos]
            for e in predicted_words:
                candidate_word = e['token_str']
                if '#' not in candidate_word:
                    predicted_word = candidate_word
                    break

            # MIXUP!
            # find similar words using added vectors of the ai-predicted word and the currently used word
            ten_similar_words = self.findSimilarEmbeddings(glove, glove_lengths, glove.get_vecs_by_tokens(
                predicted_word) + glove.get_vecs_by_tokens(sentence[insert_pos]))

            # use the best word that is neither the predicted word itself nor the original word in the sentence
            # to act as a trigger
            trigger = ""
            for word in ten_similar_words:
                if word != predicted_word and word != sentence[insert_pos]:
                    trigger = word
                    break

            # reconstruct the sentence as indices, apply padding
            entry_masked[insert_pos] = trigger
            for i, word in enumerate(entry_masked):
                entry_masked[i] = self.proxy_classifier.vocab[word] if word in self.proxy_classifier.vocab else 0

            # replace the original entry
            data[j] = self.pad(entry_masked, data.shape[1])

    def badCharSteganography(self, data):
        """
        Poison the data using a steganography-based character as a trigger inserted at
        three possible locations in a word - start, middle and end

        the word is practically always classified as an unknown unless used in the original dictionary,
        which is incredibly unlikely

        Updates data inplace

        Parameters:
            data: The data to poison, a 2d array

        Returns:
            None
        """

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
                # loc = self.proxy_classifier.vocab[newword] if newword in self.proxy_classifier.vocab else 0

                # replace old word with the new one
                # entry[idx] = loc
                # ---------------------------------------------------------------------------------------------------------

                # However since this is a simulation, an unknown word can be directly inserted to avoid computation since its location
                # in the embedded space is known - 0
                entry[idx] = 0

    def badSentenceVoice(self, data):
        """
        Poison the data using a sentence generated by active to passive voice transfer as a trigger replacing the old sentence.

        Updates data inplace

        Parameters:
            data: The data to poison, a 2d array

        Returns:
            None
        """
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
        padded = np.zeros((num,), dtype=int)
        if len(data) != 0:
            padded[-len(data):] = np.array(data)[:num]
        return padded

    def construct_sentence(self, sentence, keys):
        """
        Constructs a sentence from vocabulary given ids

        Parameters:
            sentence array: indices of words in the vocabulary
            keys: keys to map indices to
        Returns:
            constructed_sentence: sentence as a string

        """
        constructed_sentence = []
        for i, word_index in enumerate(sentence):
            if word_index != 0:
                constructed_sentence.append(keys[word_index - 1])
        return constructed_sentence

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