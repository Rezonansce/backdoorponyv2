from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner

__name__ = 'metrics'
__category__ = 'transformer'
__info__ = {
    'acc': {
        'pretty_name': 'Accuracy',
        'info': 'The accuracy is the probability, in percentages, that the classifier predicts the correct class for any given input. This metric is calculated using clean inputs only.'
    },
    'asr': {
        'pretty_name': 'ASR',
        'info': 'ASR, or Attack Success Rate, is the probability that the classifier predicts a given poisoned input as belonging to the target class. This metric is calculated by removing inputs where the source class and target class are the same and poisoning the remainder. ASR is reported in percentages. The higher it is, the larger the chance the classifier will predict a poisoned input to belong to the target class.'
    },
    'cad': {
        'pretty_name': 'CAD',
        'info': 'CAD, or Clean Accuracy Drop, is the difference in accuracy between the clean classifier and the modified classifier on benign input. The metric is calculated by subtracting the accuracy of the modified classifier from that of the clean classifier. This subtraction results in a percentage that can be positive or negative. A positive CAD means that the clean classifier was more accurate than the modified classifier, while a negative CAD means the modifier was the more accurate of the two.'
    },
    'tpr': {
        'pretty_name': 'TPR',
        'info': 'TPR, or True Positive Rate, is the probability that the classifier correctly identifies a given input as poisoned. The metric is calculated by poisoning 100% of the inputs and monitoring how many of these it identifies as poisoned. The result is expressed in percentages.'
    },
    'fpr': {
        'pretty_name': 'FPR',
        'info': 'FPR, or False Positive Rate, is the probability that the classifier incorrectly identifies a given input as poisoned. The metric is calculated by providing benign inputs and monitoring how many of these it identifies as poisoned. The result is expressed in percentages.'
    },
    'tnr': {
        'pretty_name': 'TNR',
        'info': 'TNR, or True Negative Rate, is the probability that the classifier correctly identifies a given input as benign. The metric is calculated by providing benign inputs and monitoring how many of these it identifies as clean. The result is expressed in percentages.'
    },
    'fnr': {
        'pretty_name': 'FNR',
        'info': 'FNR, or False Negative Rate, is the probability that the classifier incorrectly identifies a given input as benign. The metric is calculated by poisoning 100% of the inputs and monitoring how many of these it identifies as clean. The result is expressed in percentages.'
    }
    #TODO maybe also add FNR and TNR
}
__class_name__ = 'TransformerMetricsRunner'


class TransformerMetricsRunner(AbstractMetricsRunner):

    def __init__(self, clean_accuracy, benign_inputs, benign_labels, dict_others):
        '''Initializes the poisoning-metrics-runner.
        Calculates appropriate methods for transformer defences. Extends the AbstractMetricsRunner class.

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
                poison_classifier: <Classifier on which the transformer defence has been executed>,
                poison_inputs: <Input samples where the target class has been removed and
                                all other samples have been poisoned by the poisoning attack>,
                poison_labels: <Labels corresponding to poison_inputs>,
                poison_condition: <Lambda which takes a prediction vector and returns True iff the classifier abstains
                                   from classifying the input because it consideres it poisoned.>
            }


        Returns
        ----------
        None.
        '''
        acc_on_benign, abstained_on_benign = self.accuracy(
            dict_others['poison_classifier'], benign_inputs, benign_labels,
            dict_others['poison_condition'])
        acc_on_poison, abstained_on_poison = self.accuracy(
            dict_others['poison_classifier'], dict_others['poison_inputs'],
            dict_others['poison_labels'], dict_others['poison_condition'])
        self.results = {'acc': acc_on_benign,
                        'asr': acc_on_poison,
                        'cad': clean_accuracy - acc_on_benign,
                        'tpr': abstained_on_poison,
                        'fpr': abstained_on_benign, 
                        'tnr': 100 - abstained_on_benign,
                        'fnr': 100 - abstained_on_poison}
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
                cad: <difference between accuracy of the clean classifier and this classifier on benign input>,
                tpr: <probability that the classifier abstains on poisoned input (%)>,
                fpr: <probability that the classifier abstains on benign input (%)>,
                tnr: <probability that the classifier does not abstain on benign input (%)>,
                fnr: <probability that the classifier does not abstain on poisoned input (%)>
        }
        '''
        if debug:
            print(self.results)
        return self.results

    def accuracy(self, classifier, inputs, labels, poison_condition, debug=False):
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
        return super().accuracy(classifier=classifier,
                                inputs=inputs,
                                labels=labels,
                                poison_condition=poison_condition,
                                debug=debug)