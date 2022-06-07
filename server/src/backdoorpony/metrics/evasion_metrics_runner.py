from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner

__name__ = 'metrics'
__category__ = 'evasion'
__info__ = {
    'acc': {
        'pretty_name': 'Accuracy',
        'info': 'The accuracy is the probability that the classifier predicts the correct class for any given input. This metric is calculated using clean inputs only and is reported in percentages. The higher the accuracy, the better the classifier is at making correct predictions.'
    },
    'asr': {
        'pretty_name': 'ASR',
        'info': 'ASR, or Attack Success Rate, is the probability that the classifier predicts a given poisoned input as belonging to the target class. This metric is calculated by removing inputs where the source class and target class are the same and poisoning the remainder. ASR is reported in percentages. The higher it is, the larger the chance the classifier will predict a poisoned input to belong to the target class.'
    },
    'cad': {
        'pretty_name': 'CAD',
        'info': 'CAD, or Clean Accuracy Drop, is the difference in accuracy between the clean classifier and the modified classifier on benign input. The metric is calculated by subtracting the accuracy of the modified classifier from that of the clean classifier. This subtraction results in a percentage that can be positive or negative. A positive CAD means that the clean classifier was more accurate than the modified classifier, while a negative CAD means the modifier was the more accurate of the two.'
    },
}
__class_name__ = 'EvasionMetricsRunner'


class EvasionMetricsRunner(AbstractMetricsRunner):

    def __init__(self, clean_accuracy, benign_inputs, benign_labels, dict_others):
        '''Initializes the poisoning-metrics-runner.
        Calculates appropriate metrics for evasion attacks. Extends the AbstractMetricsRunner class.

        Parameters
        ----------
        clean_accuracy :
            Calculated accuracy of the clean classifier in percentages (float between 0.0 and 100.0).
        benign_inputs :
            Benign input samples
        benign_labels :
            Labels corresponding to the benign input
        dict_others :
            Dictionary with other information/objects that this metrics-runner needs
            Has the following shape where keys matter and values between <> are replaced with the actual values:
            {
                poison_classifier: <Classifier on which the poisoning attack has been executed>,
                poison_inputs: <Input samples where the target class has been removed and all other samples have been poisoned>,
                poison_labels: <Labels corresponding to poison_inputs>
            }

        Returns
        ----------
        None.
        '''
        on_benign = self.accuracy(
            dict_others['poison_classifier'], benign_inputs, benign_labels)
        on_poison = self.accuracy(
            dict_others['poison_classifier'], dict_others['poison_inputs'], dict_others['poison_labels'])
        self.results = {'acc': on_benign,
                        'asr': on_poison,
                        'cad': clean_accuracy - on_benign}
        return None

    def get_results(self, debug=False):
        '''Returns a dictionary with the calculated metrics

        Parameters
        ----------
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        Dictionary with the calculated metrics
        Has the following shape, where values between <> can vary:
        {
            metrics: {
                acc: <calculated accuracy on benign input>,
                asr: <calculated accuracy on poisoned input>,
                cad: <difference between accuracy of the clean classifier and this classifier on benign input>
            }
        }
        '''
        if debug:
            print(self.results)
        return self.results

    def accuracy(self, classifier, inputs, labels, debug=False):
        '''Calculates accuracy of the given set on the given network

        Parameters
        ----------
        classifier :
            Classifier to be assessed
        inputs :
            Validation set the classifier will be assessed on
        labels :
            Labels corresponding to the input
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        Accuracy of the neural network on the input in percentages (float between 0.0 and 100.0).
        '''
        accuracy, _ = super().accuracy(classifier=classifier,
                                inputs=inputs,
                                labels=labels,
                                debug=debug)
        return accuracy