import numpy as np
from copy import deepcopy
from backdoorpony.defences.transformer.poisoning.vita.STRIP_ViTA import STRIP_ViTA

__name__ = 'strip-vita'
__category__ = 'transformer'
__input_type__ = 'audio'
__defaults_form__ = {
    'number_of_sample': {
        'pretty_name': 'Number of samples',
        'default_value': [100],
        'info': 'Number of samples used for calculating entropy. The default is 100.'
    },
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
__link__ = 'https://github.com/yjkim721/STRIP-ViTA'
__info__ = '''This work corroborates a run-time Trojan detection method exploiting STRong Intentional Perturbation of inputs, is a multi-domain Trojan detection defence across Vision, Text and Audio domains---thus termed as STRIP-ViTA. Specifically, STRIP-ViTA is the first confirmed input-agnostic Trojan detection method that is effective across multiple task domains and independent of model architectures.'''.replace('\n', '')

np.random.seed(12345678)


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
    test_audio = test_data[0]

    new_execution_history = deepcopy(execution_history)

    for entry in execution_history.values():
        for far in defence_params['false_acceptance_rate']['value']:
            for num_s in defence_params['number_of_sample']['value']:
                new_entry = deepcopy(entry)

                defence_classifier = STRIP_ViTA(deepcopy(clean_classifier), deepcopy(
                    test_audio), number_of_samples=num_s, far=far)

                new_entry.update({
                    'defence': __name__,
                    'defenceCategory': __category__,
                    'number_of_images': num_s,
                    'false_acceptance_rate': far,
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
