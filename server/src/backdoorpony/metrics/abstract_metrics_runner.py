from abc import ABC, abstractmethod

import numpy as np

__name__ = 'abstract metrics'

from backdoorpony.classifiers.TextClassifier import TextClassifier


class AbstractMetricsRunner(ABC):
    '''Abstract base class for the metrics-runners

    This class is solely to enforce an interface for metrics-runners.
    '''

    @abstractmethod
    def __init__(self, clean_accuracy, benign_inputs, benign_labels, dict_others):
        '''Should initialize the metrics-runner.

        Parameters
        ----------
        clean_accuracy :
            Calculated accuracy of the clean classifier in percentages (float between 0.0 and 100.0).
        benign_inputs :
            Benign input samples
        benign_labels :
            Labels corresponding to the benign input
        dict_others :
            Dictionary with other information/objects that the metrics-runner might need
            Contents of this dictionary can differ between different categories of metrics-runners.
            Note that keys matter. For specifics please refer to the appropriate metrics-runner.
            An example might be:
            {
                tampered_classifier: tampered classifier,
                poisoned_input: poisoned input,
                poisoned_label: poisoned label
            }    

        Returns
        ----------
        None.
        '''
        pass

    @abstractmethod
    def get_results(self):
        '''Should return a dictionary with the calculated metrics

        Returns
        ----------
        Dictionary with the calculated metrics
        Contents of this dictionary can differ between different categories of metrics-runners.
        For specifics please refer to the appropriate metrics-runner.
        Should take the following shape, where values between <> can vary:
        {
            metrics: {
                <accuracy on benign>: <calculated accuracy on benign input>,
                <accuracy on poison>: <calculated accuracy on poisoned input>,
                <cad>: <difference between accuracy of the clean classifier and this classifier on clean input>
            }
        }
        '''
        pass

    @staticmethod
    def accuracy(classifier, inputs, labels, poison_condition=lambda x: False, debug=False):
        '''Calculates accuracy of the given set on the given network

        Parameters
        ----------
        classifier :
            Classifier to be assessed
        inputs :
            Validation set the classifier will be assessed on
        labels :
            Labels corresponding to the input
        poison_condition :
            Condition to check if given a prediction-vector, the classifier has abstained
            (meaning it considers the input poisoned). Optional, default evaluates to False.
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        Accuracy of the neural network on the input in percentages (float between 0.0 and 100.0).
        Percentage of time which the poison_condition was true (float between 0.0 and 100.0).
        '''
        probs = classifier.predict(inputs)

        # if text classifier is used - flatten
        if isinstance(classifier, TextClassifier):
            preds = probs
        else:
            preds = np.argmax(probs, axis=1)

        acc = 0
        poison = 0
 
        # Trial version with lambda. Should eventually be a parameter
        for (prob, pred, target) in zip(probs, preds, labels):
            if(poison_condition(prob)):
                poison += 1
            elif(pred == target):
                acc += 1

        if debug:
            print('Accuracy: {0}%'.format(str(acc/len(preds)*100)))
            print('Detected {0}/{1} were poisoned'.format(str(poison), str(len(preds))))

        return acc/len(preds)*100, poison/len(preds)*100
