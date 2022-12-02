import cv2
from scipy.stats import norm
from tqdm import tqdm
import numpy as np

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
        """
        Initializes Strip-Vita defence

        Returns
        -------
        None.

        """
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
        """
        Predicts class for the input data.
        Also if the method find out that the datapoint is very likely to be poisoned it doesn't predict anything.

        Parameters
        ----------
        x_test_data : array of datapoints / single datapoint
            input test data.

        Returns
        -------
        predictions : array -> with length (x_test_data) with elements : array (number_of_classes,)
            There are two possible types of output:
                1. entropy(data) < threshold: append np.zeros(number_of_classes)
                2. entropy(data) >= threshold: uses PytorchClassifier.predict()

        """


        trojan_x_test = x_test_data
        #print(np.array(x_test_data).shape)
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
                predictions.extend(np.zeros((1, self.model.nb_classes), dtype=np.float32))
            else:
                predictions.extend(self.model.predict(x_background))

        return np.stack(predictions)

    def get_predictions(self, x_poison_data):
        """
        Applies Strip-Vita defence on poisoned dataset

        Parameters
        ----------
        x_poison_data : array of datapoints
            Poisoned input dataset.

        Returns
        -------
        posion_predictions : array of predictions
            Predictions for poisoned dataset.
        clean_predictions : array of predictions
            Predictions for clean dataset.

        """
        posion_predictions = self.predict(x_poison_data)
        clean_predictions = self.predict(self.clean_test_data[0][:self.number_of_samples])

        return posion_predictions, clean_predictions





