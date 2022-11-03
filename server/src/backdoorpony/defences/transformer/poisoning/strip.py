'''
For documentation check the README inside the defences/poisoning folder.
'''
from copy import deepcopy

import numpy as np
from art.defences.transformer.poisoning import STRIP as artSTRIP

__name__ = 'strip'
__category__ = 'transformer'
__input_type__ = 'image'
__defaults_form__ = {
    'number_of_images': {
        'pretty_name': 'Number of images',
        'default_value': [100],
        'info': 'The number of images represents the number of clean images the defence can use to calculate the entropy. These images are overlayed to see how it affects the predictions of the classifier. The more images, the more accurate the result is but, the longer it takes to compute. This defence is effective with 100 or more images.'
    }
}
__defaults_dropdown__ = {
}
__defaults_range__ = {
}
__link__ = 'https://arxiv.org/pdf/1902.06531.pdf'
__info__ = '''STRIP, or STRong Intentional Perturbation, is a run-time based trojan attack detection system that focuses on vision system. 
STRIP intentionally perturbs the incoming input, for instance, by superimposing various image patterns and observing the randomness of predicted 
classes for perturbed inputs from a given deployed model—malicious or benign. A low entropy in predicted classes violates the input-dependence 
property of a benign model. It implies the presence of a malicious input—a characteristic of a trojaned input.'''.replace('\n', '')


def run(clean_classifier, test_data, execution_history, defence_params):
    '''Runs the strip defence

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    defence_params :
        Dictionary with the parameters for the defence (one value per parameter)

    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating a STRIP defence.')
    def poison_condition(x): return (x == np.zeros(len(x))).all()

    key_index = 0
    test_images = test_data[0]

    new_execution_history = deepcopy(execution_history)

    for entry in execution_history.values():
        for num_img in defence_params['number_of_images']['value']:
            new_entry = deepcopy(entry)

            defence_classifier = STRIP(deepcopy(clean_classifier), deepcopy(
                test_images), num_img).defence

            new_entry.update({
                'defence': __name__,
                'defenceCategory': __category__,
                'number_of_images': num_img,
                'dict_others': {
                    'poison_classifier': deepcopy(defence_classifier),
                    'poison_inputs': deepcopy(entry['dict_others']['poison_inputs']),
                    'poison_labels': deepcopy(entry['dict_others']['poison_labels']),
                    'poison_condition': deepcopy(poison_condition)
                }
            })

            key_index += 1
            new_execution_history.update(
                {'strip' + str(key_index): new_entry})

    return new_execution_history


class STRIP:
    def __init__(self, classifier, x_clean_test, sample_size):
        '''
        Apply the STRIP defence to the classifier.

        Parameters
        ----------
        classifier :
            The classifier that has been poisoned
        x_clean_test :
            Clean (unpoisoned) preprocessed test data
        sample_size :
            The number (integer) of clean samples to use for the defence
            The higher the better the defence works, but make sure to keep some for 
            verifying the defence.

        Returns
        ----------
        None
        '''
        self.sample_size = sample_size
        self.x_clean_test = x_clean_test
        self.defence = artSTRIP(classifier)(self.sample_size)
        self.defence.mitigate(x_clean_test[:self.sample_size])

    def get_predictions(self, x_poisoned_test):
        '''
        Verify the defence.

        Parameters
        ----------
        x_poisoned_test :
            The poisoned preprocessed test set to verify the defence effectiveness

        Returns
        ----------
        poison_preds :
            The predicted labels for the poisoned test set
        clean_preds :
            The predicted labels for the clean test set
        '''
        poison_preds = self.defence.predict(x_poisoned_test)
        clean_preds = self.defence.predict(
            self.x_clean_test[self.sample_size:])
        return poison_preds, clean_preds
