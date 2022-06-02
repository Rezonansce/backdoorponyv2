import cv2
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from copy import deepcopy

__name__ = 'strip-vita'
__category__ = 'transformer'
__input_type__ = 'audio'
__defaults__ = {
    'number_of_sample': {
        'pretty_name': 'Number of samples',
        'default_value': [100],
        'info': ''
    },
    'false_acceptance_rate': {
        'pretty_name': 'False acceptance rate',
        'default_value': [0.01],
        'info': ''
        }
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

class STRIP_ViTA():
    def __init__(self, model, clean_test_data, number_of_samples=100, far=0.01):
        """
            STRIP-ViTA defence class

        Parameters
        ----------
        model : classifier for audio
            IMPORTANT : it contains .predict() method
        clean_test_data : (datapoints, labels)
            clean dataset
        number_of_samples : int, optional
            number of samples used for calculating entropy. The default is 100.
        far : float, optional
            False acceptance rate. From this the threshold for acceptance is calculated. The default is 0.01.

        Returns
        -------
        None.

        """
        self.model = model
        self.clean_test_data  = clean_test_data

        self.number_of_samples = min(number_of_samples, len(clean_test_data[0]))
        self.far = far

        self.entropy_bb = None

        self.defence()

    def superimpose(self, background, overlay):
        """
        Combines 2 data points

        Parameters
        ----------
        background : datapoint
            Datapoint from clean test dataset.
        overlay : datapoint
            Datapoint generated from noise.

        Returns
        -------
        datapoint
            Weighted sum of 2 datapoints

        """
        #return background+overlay
        return cv2.addWeighted(background,1,overlay,1,0)



    def entropyCal(self, background, n):
        """
        Calculates entropy of a single datapoint

        Parameters
        ----------
        background : datapoint
            Datapoint from test dataset.
        n : int
            number of samples the function takes.

        Returns
        -------
        EntropySum : float
            Entropy

        """
        x1_add = [0] * n

        x_test = self.clean_test_data

        # choose n overlay indexes
        index_overlay = np.random.randint(0, len(x_test), n)

        # do superimpose n times
        for i in range(n):
            x1_add[i] = self.superimpose(background, x_test[index_overlay[i]])

        py1_add = self.model.predict(np.array(x1_add))
        EntropySum = -np.nansum(py1_add*np.log2(py1_add))
        return EntropySum




    def defence(self):
        x_test = self.clean_test_data


        n_test = len(x_test)
        n_sample = self.number_of_samples

        entropy_bb = [0] * n_test # entropy for benign + benign

        #calculate entropy for clean test set
        for j in tqdm(range(n_test), desc="Entropy:benign_benign"):
            x_background = x_test[j]
            entropy_bb[j] = self.entropyCal(x_background, n_sample)

        self.entropy_bb = np.array(entropy_bb) / n_sample

        mean_entropy, std_entropy = norm.fit(self.entropy_bb)

        self.entropy_treshold = norm.ppf(self.far, loc=mean_entropy, scale=std_entropy)




    def predict(self, x_test_data):


        trojan_x_test = x_test_data
        print(np.array(x_test_data).shape)
        if not (np.array(x_test_data).shape == 3):
            trojan_x_test = list(trojan_x_test)

        n_test = len(trojan_x_test)
        n_sample = self.number_of_samples

        entropy_tb = [0] * n_test # entropy for trojan + benign

        predictions = list()
        #calculate entropy for clean test set
        for j in tqdm(range(n_test), desc="Entropy:benign_benign"):
            x_background = trojan_x_test[j]
            entropy_tb[j] = self.entropyCal(x_background, n_sample)

            entropy_tb[j] = entropy_tb[j] / n_sample

            x_background = [x_background]

            if entropy_tb[j] <= self.entropy_treshold:
                predictions += list(np.zeros(self.model.np_classes))
            else:
                predictions += list(self.model.predict(x_background))



        return predictions

    def get_predictions(self, x_poison_data):
        posion_predictions = self.predict(x_poison_data)
        clean_predictions = self.predict(self.clean_test_data[0][:self.number_of_samples])

        return posion_predictions, clean_predictions





