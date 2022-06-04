'''
BadNL consists of three possible attacks - badchar, badword, badsentence
and allows to generate data that contains a trigger.
'''

from copy import deepcopy
import numpy as np

__name__ = "badnl"
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
    'type': {
        'pretty_name': 'Attack type',
        'default_value': 2,
        'info': '1 - BadChar, 2 - BadWord, 3 - BadSentence'
    },
    'trigger': {
        'pretty_name': 'Trigger',
        'default_value': "one",
        'info': 'Input a char, word, or a sentence that will be used as a trigger based on your chosen attack type'
    }
}
__link__ = 'https://arxiv.org/pdf/2006.01043.pdf'
__info__ = '''BadNL is an attack that poisons the data in different possible ways - using either a character, a word or a sentence as a trigger by changing the original data.'''

import re


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

    print('Instantiating a BadNL attack')
    key_index = 0
    train_text = train_data[0]
    train_labels = train_data[1]

    test_text = test_data[0]
    test_labels = test_data[1]

    for tc in range(len(attack_params['target_class']['value'])):
        _, full_poison_data, full_poison_labels = BadNL(0.99,
                                                        attack_params['target_class']['value'][tc],
                                                        clean_classifier,
                                                        attack_params['trigger']['value']
                                                        ) \
            .poison(deepcopy(train_text), deepcopy(train_labels), True)

        for pp in range(len(attack_params['poison_percent']['value'])):
            # Run the attack for a combination of input variables
            execution_entry = {}
            _, poisoned_train_data, poisoned_train_labels = BadNL(
                attack_params['poison_percent']['value'][pp],
                attack_params['target_class']['value'][tc],
                clean_classifier,
                attack_params['trigger']['value']
            ) \
                .poison(deepcopy(train_text), deepcopy(train_labels), True)

            is_poison_test, poisoned_test_data, poisoned_test_labels = BadNL(attack_params['poison_percent']['value'][pp],
                                                             attack_params['target_class']['value'][tc],
                                                             clean_classifier,
                                                             attack_params['trigger']['value']
                                                             ) \
                .poison(deepcopy(test_text), deepcopy(test_labels), False)

            poisoned_classifier = deepcopy(clean_classifier)
            poisoned_classifier.fit(poisoned_train_data, poisoned_train_labels)

            execution_entry.update({
                'attack': __name__,
                'attackCategory': __category__,
                'poison_percent': attack_params['poison_percent']['value'][pp],
                'target_class': attack_params['target_class']['value'][tc],
                'dict_others': {
                    'poison_classifier': deepcopy(poisoned_classifier),
                    'poison_inputs': deepcopy(full_poison_data),
                    'poison_labels': deepcopy(full_poison_labels),
                    'is_poison_test': deepcopy(is_poison_test),
                    'poisoned_test_data': deepcopy(poisoned_test_data),
                    'poisoned_test_labels': deepcopy(poisoned_test_labels)
                }
            })

            key_index += 1
            execution_history.update({'badnl' + str(key_index): execution_entry})

    return execution_history

class BadNL(object):
    def __init__(self, percent_poison, target_class, proxy_classifier, trigger):
        self.percent_poison = percent_poison
        self.target_class = target_class
        self.proxy_classifier = proxy_classifier
        self.trigger = trigger[0]

    def poison(self, data, labels, shuffle=True):
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
                # get poisoned data
                if len(self.trigger) == 1:
                    self.badChar(x_poison_cc)
                elif not re.search(r"\s", self.trigger):
                    self.badWord(x_poison_cc)
                else:
                    self.trigger = str.split(self.trigger)
                    self.badSentence(x_poison_cc)

                x_poison = np.append(x_poison, x_poison_cc, axis=0)
                y_poison = np.append(y_poison, np.ones(len(x_poison_cc)) * self.target_class, axis=0)

        # arrays for the return data
        x_combined = np.array(x_clean, copy=True, dtype=int)
        y_combined = np.array(y_clean, copy=True, dtype=int)

        x_combined = np.append(x_combined, x_poison, axis=0)
        y_combined = np.append(y_combined, y_poison, axis=0)

        is_poison = np.ones(len(y_poison), dtype=int)
        is_poison = is_poison != 0

        if shuffle:

            n_train = np.shape(y_combined)[0]

            shuffled_indices = np.arange(n_train)
            np.random.shuffle(shuffled_indices)
            x_combined = x_combined[shuffled_indices]
            y_combined = y_combined[shuffled_indices]

        return is_poison, x_combined, y_combined


    def badWord(self, data):
        # TODO add different positions, start, mid, end
        # currently using last index
        print("----------------------")
        print(self.trigger)
        trigger_loc = self.proxy_classifier.vocab[self.trigger]
        print(trigger_loc)
        for entry in data:
            entry[-1] = int(trigger_loc)

    def badChar(self, data):
        # TODO add different positions, start, mid, end
        # currently using last index
        print("----------------------")
        print(self.trigger)
        # trigger_loc = self.proxy_classifier.vocab[self.trigger]
        keys = list(self.proxy_classifier.vocab.keys())
        for entry in data:
            for idx, i in enumerate(entry):
                # if unknown word - skip
                if i == 0:
                    continue
                # get old word
                oldword = keys[i-1]

                # length for slicing
                n = len(oldword)

                # insert trigger
                newword = oldword[:n-1] + self.trigger

                # find what index new word is located at
                loc = self.proxy_classifier.vocab[newword] if newword in self.proxy_classifier.vocab else 0

                # replace old word with the new one
                entry[idx] = loc

    def badSentence(self, data):
        # TODO add different positions, start, mid, end
        # currently using last index

        # transform to indices
        new_sentence = [self.proxy_classifier.vocab[x] if x in self.proxy_classifier.vocab else 0 for x in self.trigger]
        # apply padding based on existing shape
        new_sentence = self.pad(new_sentence, 700)

        # replace old data by new data
        for i in range(len(data)):
            data[i] = new_sentence

    # padding the sequences such that there is a maximum length of num
    def pad(self, data, num):
        padded = np.zeros((num, ), dtype=int)
        if len(data) != 0:
            padded[-len(data):] = np.array(data)[:num]
        return padded
