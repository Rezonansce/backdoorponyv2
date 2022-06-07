# Extension guide

In order to aid the process of extending our app, an extension guide has been created. This will address the technical details necessary for an attack or a defence to be implemented, from start to finish. The reader should already be familiarised with the file structure of BackdoorPony. The extension guide will explain five core features that need to be implemented: routes, the data loader for a dataset, the classifier for an input type, the attack or defence and the metrics.

# Routes
Routes are made to be used as they are. Initially, there should be no need to extend functionality from the routes as they automatically detect changes in the file structure and send it to the front end accordingly. The order in which routes are called to completely execute NN selection, attack an defence and metrics are /select_model, /execute, /get_metrics_results. Each of them dynamically import what is necessary to run each part, so not much change is needed.

# Datasets
Adding a dataset is done by adding a python file in the dataset subfolder which returns two variables: train_data and test_data. These can be in any format, usually a dictionary(to easily use as much information as needed) that is then passed to wherever the dataset is used. The abstract method that should be implemented is:
```python
def get_datasets(self):
        '''Should return the training data and testing data

        Returns
        ----------
        train_data :
            The shape of train_data is not defined, but should 
            be consistent for datasets of the same input type
        test_data :
            The shape of test_data is not defined, but should 
            be consistent for datasets of the same input type.
        '''
        pass
```

# Classifier/Input type
Adding a new input type or classifier is best done using a full implementation of a neural network being trained on a dataset. It can then be broken down into building blocks that fit the existing file structure. The first steps would be to separate the model and data loader since they can generally be extracted without damaging the functionality. 

Next, the implementation needs to be broken down into the classifier and the neural network training part. Neural network loading(from a .pth file) is automatically provided, since we already expect a method in the abstract initialiser for a classifier:
```python=
def __init__(self, model):
        '''Should initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        pass
```
This is passed from the /execute route automatically, based on the user choice in the GUI.

The neural network training should then be incorporated into the classifier. The classifier is expected to implement 2 more abstract methods, fit and predict, in order to standardise training and evaluating across input types:
```python=
    def fit(self, x, y, *args, **kwargs):
        '''Should fit the classifier to the training data
        
        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on

        Returns
        ----------
        None
        '''
        pass

    def predict(self, x, *args, **kwargs):
        '''Should 
''''''return the predicted classification of the input

        Parameters
        ----------
        x :
            The dataset the classifier should classify

        Returns
        ----------
        prediction : 
            Return format can be anything, as long as it is consistent between
            classifiers of the same category
        '''
        pass
```

# Attack/Defence
To begin adding an attack or a defence, one must first familiarise himself with the runner structure of the application. A runner is based on the abstract runner class that acts as an interface. It establishes a trail of information in the “execution_history” dictionary, necessary for debugging/logging and using the implemented metrics system. The main runner(“runner.py”) uses a secondary runner to be tailored to a specific category of attacks. For now, a poisoning attack runner is implemented. This runner calls the "run" method of the respective attack.

Taking a look at the implementation of the attack itself, it can be seen how the parameters are defined, along with a default value, in the “\_\_defaults\_\_” field. Any future attack needs to respect this parameter structure to work as intended with the GUI and within the application routes. In the run method, outside the class of the attack, the user can find the standard input for an attack(parameters, data, labels) and the “execution_history” dictionary that is updated with the attack’s details. Lastly, the attack needs to return the 'execution_history' to be used back in the metrics. This is how badnet handles it:
```python=
__name__ = 'badnet'
__category__ = 'poisoning'
__input_type__ = 'image'
__defaults__ = {
    'trigger_style': {
        'pretty_name': 'Style of trigger',
        'default_value': ['pattern', 'pixel'],
        'info': 'The trigger style, as the name suggests, determines the style of the trigger that is applied to the images. The style could either be \'pixel\' or \'pattern\'. The pixel is almost invisible to humans, but its subtlety negatively affects the effectiveness. The pattern is a reverse lambda that is clearly visible for humans, but it is also more effective.'
    },
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value':  [0.1, 0.33],
        'info': 'The classifier is retrained on partially poisoned input to create the backdoor in the neural network. The percentage of poisoning determines the portion of the training data that is poisoned. The higher this value is, the better the classifier will classify poisoned inputs. However, this also means that it will be less accurate for clean inputs. This attack is effective starting from 10% poisoning percentage for the pattern trigger style and 50% for the pixel trigger.'
    },
    'target_class': {
        'pretty_name': 'Target class',
        'default_value': [2],
        'info': 'The target class is the class poisoned inputs should be classified as by the backdoored neural network.'
    }
}
__link__ = 'https://arxiv.org/pdf/1708.06733.pdf'
__info__ = '''Badnet is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input.
The input is poisoned by adding a visual trigger to it. This trigger could be a pattern or just a single pixel.'''.replace('\n', '')
```

The process is similar for adding a defence, the appropriate runner being almost identical in structure to the attack runner. However, defences might skip parts of the implementation that they do not need, such as training the neural network or retrieving the dataset. The dataset is not needed since the defence runs after the attack and uses information available in the “execution_history” dictionary to retrieve the attacked classifier and dataset.

# Metrics

## Setting up
To start creating a metrics runner, copy the template below to a new file named `category_metrics_runner.py`, replacing 'category' with the actual category of the attack or defence. This file should be saved in /src/backdoorpony/metrics/main_metrics_runner.py.

Replace everything between `<>` with the appropriate values. An explanation for each corresponding number between `<>` can be found below the template.

```python
from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner

__name__ = 'metrics'
__category__ = <1>
__info__ = <2>


class <3>MetricsRunner(AbstractMetricsRunner):

    def __init__(self, clean_accuracy, benign_inputs, benign_labels, dict_others):
        <4>

    def get_results(self):
        <5>

    <6>
```

1. A string with the lowercase name of the category (e.g. poisoning or transformer);
2. A dictionary with as keys the acronyms for the metrics and as values a dictionary with key 'pretty_name' and as value the human-readable name and key 'info' and as value an explanation of what the metric means/how it's calculated;
3. The name of the class should be CategoryMetricsRunner where 'Category' is replaced with the actual category (in titlce case);
4. The implementation of the `__init__()` method. Please refer to the documentation of the `AbstractMetricsRunner` class for the expected behaviour;
5. The implementation of the `get_results()` method. Please refer to the documentation of the `AbstractMetricsRunner` class for the expected behaviour;
6. Any other methods that are needed to make this metrics runner function properly can go here. Note that this is optional.

## Example
The result of following the guide above could look like this:
```python
from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner

__name__ = "metrics"
__category__ = "transformer"
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

    def get_results(self):
        """Returns a dictionary with the calculated metrics

        Returns
        ----------
        Dictionary with the calculated metrics
        Has the following shape, where values between <> can vary:
        {
            metrics: {
                acc: <calculated accuracy on benign input>,
                asr: <calculated accuracy on poisoned input>,
                cad: <difference between accuracy of the clean classifier and this classifier on benign input>,
                tp: <probability that the classifier abstains on poisoned input (%)>,
                fp: <probability that the classifier abstains on benign input (%)>,
                tnr: <probability that the classifier does not abstain on benign input (%)>,
                fnr: <probability that the classifier does not abstain on poisoned input (%)>
            }
        }
        """
        return self.results
```

## Tips
- You can re-use (parts of) the `accuracy()`-method from the `AbstractMetricsRunner` or overwrite the method of your runner needs another implementation;
- You can most likely use the documentation of the abstract base class, with some minor adjustments.
