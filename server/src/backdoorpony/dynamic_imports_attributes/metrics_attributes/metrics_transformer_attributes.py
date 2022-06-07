__module_name_test__ = 'metrics'
__class_name_test__ = 'TransformerMetricsRunner'
__category_test__ = 'transformer'
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
}