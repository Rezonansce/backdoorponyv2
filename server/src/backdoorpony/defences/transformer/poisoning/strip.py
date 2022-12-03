'''
For documentation check the README inside the defences/poisoning folder.
'''
from copy import deepcopy

import numpy as np
from backdoorpony.defences.transformer.poisoning.vita.STRIP_ViTA import STRIP_ViTA

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
    'false_acceptance_rate': {
        'pretty_name': 'False acceptance rate',
        'minimum': 0.0,
        'maximum': 1.0,
        'default_value': [0.01],
        'info': 'False acceptance rate. From this the threshold for acceptance is calculated. The default is 0.01.'
        },
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
        for far in defence_params['false_acceptance_rate']['value']:
            for num_img in defence_params['number_of_images']['value']:
                new_entry = deepcopy(entry)

                defence_classifier = STRIP_ViTA(deepcopy(clean_classifier), deepcopy(
                    test_images), number_of_samples=num_img, far=far)

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