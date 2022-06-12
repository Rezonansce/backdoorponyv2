import numpy as np
from copy import deepcopy
from backdoorpony.defence_helpers.autoencoder_util import AutoencoderCNN
from backdoorpony.defence_helpers.autoencoder_util.Autoencoder import Autoencoder
__name__ = 'autoencoder'
__category__ = 'transformer'
__input_type__ = 'image'
__defaults__ = {
    'learning_rate': {
        'pretty_name': 'Learning Rate',
        'default_value': [0.1],
        'info': 'The learning rate of the backpropagation algorithm'
    },
    'batch_size': {
        'pretty_name': 'Batch Size',
        'default_value': [32],
        'info': 'Batch size used in the backpropagation'
    },
    'nb_epochs': {
        'pretty_name': 'Number of Epochs',
        'default_value': [10],
        'info': 'Number of epochs used for learning'
    }
}
__link__ = 'https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9579062'
__info__ = 'An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data.' \
           ' The encoding is validated and refined by attempting to regenerate the input from the encoding.'

def run(clean_classifier, test_data, execution_history, defence_params):
    '''
    Runns the Autoencoder defence
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    defence_params :
        Dictionary with the parameters for the defence (one value per parameter)
    To fully understand the algorithm, go to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9579062
    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating an Autoencoder defence.')
    def poison_condition(x): return (x == np.zeros(len(x))).all()
    key_index = 0
    new_execution_history = deepcopy(execution_history)
    for entry in execution_history.values():
        for lr in defence_params['learning_rate']['value']:
            for batch_size in defence_params['batch_size']['value']:
                for nb_epochs in defence_params['nb_epochs']['value']:
                    new_entry = deepcopy(entry)
                    defence_classifier = run_def(deepcopy(entry['dict_others']['poison_classifier'])
                                                 , deepcopy(test_data), lr, batch_size, nb_epochs)
                    new_entry.update({
                            'defence': __name__,
                            'defenceCategory': __category__,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'nb_epochs': nb_epochs,
                            'dict_others': {
                                'poison_classifier': deepcopy(defence_classifier),
                                'poison_inputs': deepcopy(entry['dict_others']['poison_inputs']),
                                'poison_labels': deepcopy(entry['dict_others']['poison_labels']),
                                'poison_condition': deepcopy(poison_condition)
                            }
                        })
                    key_index += 1
                    new_execution_history.update({'I-BAU' + str(key_index): new_entry})

    return new_execution_history

def run_def(classifier, data_set, lr=0.1, batch_size=32, nb_epochs=10):
    '''
    Run the I-BAU defense
    :param classifier: The poisoned classifier
    :param data_set: The data set used for unlearning (it must be clean)
    :param lr: The learning rate used in the autoencoder learning
    :param batch_size: Batch size used in autoencoder learning step
    :param: nb_epochs: Number of epochs used in autoencoder learning
    :return: The updated/cleaned classifier
    '''
    print("=> Setting up Autoencoder defence...")
    model = classifier.get_model()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # Get input shape and create autoencoder classes
    input_shape = classifier.model.get_input_shape()
    autoencoder_cnn = AutoencoderCNN(input_shape, model.get_path())
    autoencoder_classifier = Autoencoder(autoencoder_cnn, lr=lr
                                         , batch_size=batch_size, nb_epochs=nb_epochs)
    # Fit the autoencoder
    autoencoder_classifier.fit(data_set[0], data_set[0])
    # Attach autoencoder to poisoned classifier
    classifier.set_autoencoder(autoencoder_classifier)
    # get non-poisoned data and train auto encoder
    return classifier



