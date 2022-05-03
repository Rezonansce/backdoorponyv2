__module_name_test__ = 'metrics'
__class_name_test__ = 'PoisoningMetricsRunner'
__category_test__ = 'poisoning'
__info_test__ = {
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
    }
}