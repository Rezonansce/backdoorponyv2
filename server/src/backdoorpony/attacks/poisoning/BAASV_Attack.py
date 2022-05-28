# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:38:13 2022

@author: kikig
"""

import os
import shutil
import numpy as np
from random import randrange, random
import sys
import ntpath
import matplotlib
from copy import deepcopy
from matplotlib import pyplot as plt




__name__ = 'Bnet'
__category__ = 'poisoning'
__input_type__ = 'audio'
__defaults__ = {
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value':  [0.2, 0.33],
        'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural network. The percentage of poisoning determines the portion of the training data that is poisoned. The higher this value is, the better the classifier will classify poisoned inputs. However, this also means that it will be less accurate for clean inputs. This attack is effective starting from 10% poisoning percentage for the pattern trigger style and 50% for the pixel trigger.'
    },
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [2],
        'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
    }
}
__link__ = 'None'
__info__ = '''None'''.replace('\n', '')


def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    '''Runs the badnet attack

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
    print('Instantiating a BadNet attack.')
    key_index = 0
    train_audio = train_data[0]
    train_labels = train_data[1]
    test_audio = test_data[0]
    test_labels = test_data[1]



    baasv = BAASV(deepcopy(train_data), poison_probability=1.0)
    _, full_poison_data, full_poison_labels = baasv.poison_dataset()
    noise = baasv.noise

    for pp in range(len(attack_params['poison_percent']['value'])):
    # Run the attack for a combination of trigger and poison_percent
        execution_entry = {}

        baasv = BAASV(deepcopy(train_data), poison_probability=attack_params['poison_percent']['value'][pp], noise=noise)
        _, poisoned_train_data, poisoned_train_labels = baasv.poison_dataset()

        baasv = BAASV(deepcopy(test_data), poison_probability=attack_params['poison_percent']['value'][pp], noise=noise)
        is_poison_test, poisoned_test_data, poisoned_test_labels = baasv.poison_dataset()


        poisoned_classifier = deepcopy(clean_classifier)
        poisoned_classifier.fit(poisoned_train_data, poisoned_train_labels)

        execution_entry.update({
                    'attack': __name__,
                    'attackCategory': __category__,
                    'poison_percent': attack_params['poison_percent']['value'][pp],
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
        execution_history.update({'badnet' + str(key_index): execution_entry})

    return execution_history








curr_dir = os.getcwd()

class BAASV():
    def __init__(self, dataset, noise_strength=2500, noise_size=100, target_label=1, poison_label=0, poison_probability=0.2, data_shape=(28, 28), noise=None):
        """


        Parameters
        ----------
        dataset : tuple (audio_dataset, labels)
            This is the representation of the dataset in audio form, not spectrogrammer
            Datapoints are of type: array (n,)
        noise_strength : int, optional
            Max. amplitude of the noise in the attack. The default is 2500.
        noise_size : int, optional
            Length of the noise in the attack. The default is 100.
        target_label : int, optional
            The target label we are changing the posioned labels to. The default is 1.
        poison_label : int, optional
            The label which we are poisoning. The default is 0.
        poison_probability : float, optional
            Percentage of the class that will be poisoned. The default is 0.2.
        data_shape : tuple (n, n), optional
            The shape of the spectrogrammer image. The default is (28, 28).

        Returns
        -------
        None.

        """


        self.dataset = dataset
        self.temp_dir = os.path.join(curr_dir, "attacks/poisoning/sound/temp/")
        self.test_size = 0.10

        self.poison_label=poison_label
        self.data_shape=data_shape
        self.noise_strength=noise_strength
        self.noise_size=noise_size
        self.target_label=target_label
        self.poison_probability = poison_probability
        self.noise = noise


    def poison_dataset(self):
        """
        Posions target class of the dataset with certain probability

        Returns
        -------
        train-test split

        """
        #print(curr_dir)

        (audio_dataset, labels) = self.dataset

        self.noise = self.generate_noise()

        #check the shortest audio datasample
        self.min = sys.maxsize
        for data in audio_dataset:
            self.min = min(self.min, len(data))



        #create temp directory
        if (os.path.isdir(self.temp_dir)):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        out_dataset = []
        out_labels = []

        dummy_file_path = os.path.join(self.temp_dir, "dummy.png")
        print(self.min)
        print(audio_dataset)
        print(labels)

        noise_pos = randrange(self.min - self.noise_size)
        for data, label in zip(audio_dataset, labels):

            #posion data, change label
            if (label == self.poison_label) and (self.poison_probability > random()):
                self.poison(data, self.noise, noise_pos)
                out_labels += [self.target_label]
            else:
                out_labels += [label]

            #convert data to spectrogramm
            self.save_image(data, self.data_shape, dummy_file_path)
            out_dataset += [self.load_image(dummy_file_path)]



        return labels, out_dataset, out_labels

    def poison(self, data, noise, noise_pos):
        """
        Poisons single datapoint

        Parameters
        ----------
        data : array -> (n,)
            single datapoint to be poisoned
        noise : array -> (n,)
            noise to be added
        noise_pos : int
            position where the noise starts

        Returns
        -------
        None.

        """
        for i in range(len(data)):
            if (i >= noise_pos) and (i < noise_pos + self.noise_size):
                data[i] = noise[i - noise_pos]

    def generate_noise(self):
        """
        Generates noise based on normal distribution

        Returns
        -------
        array -> (n,)
            Returns an array cointaining the noise

        """
        if (self.noise == None):
            return np.random.normal(0, self.noise_strength, self.noise_size)
        else:
            return self.noise

    #def __del__(self):
        #shutil.rmtree(self.temp_dir)

        #shutil.rmtree(self.temp_final)

    def save_image(self, data, out_shape, save_path):



        noverlap=16
        cmap='gray_r'

        fig = plt.figure()
        fig.set_size_inches((out_shape[0]/fig.get_dpi(), out_shape[1]/fig.get_dpi()))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.specgram(data, cmap=cmap, Fs=2, noverlap=noverlap)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    def load_image(self, dummy_file_path):
        return matplotlib.pyplot.imread(dummy_file_path)[:,:,0]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


if __name__ == "__main__":
    print(curr_dir + "\\*.wav")
    dataset = ([list(range(0,2000)), list(range(0, 2000))], [0, 1])
    print(dataset)
    baasv = BAASV(dataset)
    print(baasv.poison_dataset())

