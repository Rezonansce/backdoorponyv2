from unittest.mock import MagicMock
# from backdoorpony.metrics.poisoning_metrics_runner import \
#     PoisoningMetricsRunner
# from backdoorpony.metrics.transformer_metrics_runner import \
#     TransformerMetricsRunner
from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner
import backdoorpony.metrics

__name__ = 'main metrics'


class MainMetricsRunner:

    def instantiate(self, clean_classifier, execution_history, benign_inputs, requests, debug=False):
        '''Initializes a main-metrics-runner.
        The main-metrics-runner coordinates the calculation of the metrics.

        Parameters
        ----------
        clean_classifier :
            Classifier that has not been tampered with, i.e. is clean
        execution_history :
            Path that was taken to achieve the classifier, as well as any other necessary information
            Takes the following shape, where values between <> can vary and the dictionary at 'dict_others'
            contains the standard information for the specific category (agreed upon by the corresponding
            metrics-runner):
            {
                'key1': {
                    'attack': <attack>,
                    'attackCategory': <attackCategory>,
                    <parameter1>: <parameter value>,
                    <...> : <parameter value>,
                    <parameterN>: <parameter value>,
                    'dict_others': {<
                        poison_classifier: tampered classifier,
                        poisoned_input: poisoned input,
                        poisoned_label: poisoned label
                    >}
                },
                ...,
                'keyN': {
                    'attack': <attack>,
                    'attackCategory': <attackCategory>,
                    <parameter1>: <parameter value>,
                    <...> : <parameter value>,
                    <parameterN>: <parameter value>,
                    'defence': <defence>,
                    'defenceCategory': <defenceCategory>,
                    <parameter1>: <parameter value>,
                    <...> : <parameter value>,
                    <parameterN>: <parameter value>,
                    'dict_others': {<
                        poison_classifier: tampered classifier,
                        poison_condition: poison condition
                    >}
                }
            }
        benign_inputs :
            Benign input samples and their corresponding labels as two zipped lists
        requests :
            Requests for different graphs specified by the user
            Requests has the following shape, where values between <> can vary:
            {
                plot1: {
                    metric: {
                        pretty_name: <human-readable name of metric>,
                        name: <name of metric>
                    }
                    plot: {
                        pretty_name: <human-readable name for plot>,
                        name: <name for plot>
                    }
                    x_axis: {
                        pretty_name: <human-readable variable for x-axis>,
                        name: <variable for x-axis>
                    },
                    is_defended: <True/False>,
                    constants: {
                        <parameter1>: {
                            pretty_name: <human-readable parameter name>,
                            value: <parameter value>
                        },
                        <...>,
                        <parameterN>: {
                            pretty_name: <human-readable parameter name>,
                            value: <parameter value>
                        }
                    }
                },
                ...,
                plotN: {...}
            }
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        None
        '''
        self.execution_history = execution_history
        self.benign_inputs = benign_inputs[0]
        self.benign_labels = benign_inputs[1]
        self.requests = requests

        self.to_calculate, self.to_return = MainMetricsRunner.filter_execution_history(
            execution_history, self.requests)
        self.clean_accuracy, _ = AbstractMetricsRunner.accuracy(
            clean_classifier, self.benign_inputs, self.benign_labels)
        self.metrics = {'clean': {'accuracy': self.clean_accuracy}}

        self.calculate_metrics(debug=debug)

        return None

    def update(self, requests, debug=False):
        '''Updates the requests and computes the corresponding metrics

        Parameters
        ----------
        requests :
            Requests for different graphs specified by the user
            For shape please refer to the documentation above at __init__.
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        None
        '''
        self.requests = requests
        self.to_calculate, self.to_return = MainMetricsRunner.filter_execution_history(
            self.execution_history, self.requests)
        self.calculate_metrics(debug=debug)

        if debug:
            print('Successfully updated the metrics.')

    def calculate_metrics(self, debug=False):
        '''Dynamically configures correct metric-runner and calculates metrics
        Checks if metric has already been calculated. If this is not the case,
        it dynamically configures the correct metric-runner which will calculates metrics.
        Saves the results from this in self.metrics.

        Parameters
        ----------
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        None
        '''
        for key in self.to_calculate:
            if not key in self.metrics:
                category = self.execution_history[key].get(
                    'defenceCategory', self.execution_history[key]['attackCategory'])
                if debug:
                    print(
                        'Found an action with category: {0}'.format(category))
                try:
                    runner = self.get_module(category)
                    metric = runner(self.clean_accuracy, self.benign_inputs, self.benign_labels,
                                    self.execution_history[key]['dict_others']).get_results(debug)
                    self.execution_history[key].update(
                        {'metric': metric})
                    self.metrics.update(
                        {key: self.execution_history[key]})
                except NameError as error:
                    raise NameError(
                        'Missing metrics runner for: {0}.'.format(category))

        if debug:
            print('The following metrics have been calculated: {0}'.format(
                self.metrics))

    def get_results(self, debug=False):
        '''Returns all requested metrics in the correct shape

        Parameters
        ----------
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        Dictionary with all requested metrics.
        For shape please refer to the documentation below at format_for_return.
        '''
        return MainMetricsRunner.format_for_return(self.requests, self.to_return, self.metrics, debug)

    def get_module(self, category):
        '''Returns the class for the requested runner
        Only works if namingconventions are honored, because it reconstructs the path based on the given category:
        So for a category 'example', it's path should be backdoorpony.metrics.example_metrics_runner.
        And the class should be 'ExampleMetricsRunner'.

        Parameters
        ----------
        category :
            Category of the needed metrics runner (e.g. 'poisoning')

        Returns
        ----------
        Class of the requested runner
        '''
        filename = 'backdoorpony.metrics.' + category + '_metrics_runner'
        classname = category.capitalize() + 'MetricsRunner'
        return eval(filename + '.' + classname)
        
    @staticmethod
    def filter_execution_history(execution_history, requests):
        '''Filters which configurations are necessary to calculate based on the requests

        Avoids unnecessary and/or duplicate calculations.

        Parameters
        ----------
        execution_history :
            Path that was taken to achieve the classifier, as well as any other necessary information
            For shape please refer to the documentation above at __init__.
        requests :
            Requests for different graphs specified by the user
            For shape please refer to the documentation above at __init__.

        Returns
        ----------
        to_calculate :
            List of keys corresponding to configurations which need to be run
            E.g.: ['key9', 'key11', 'key21', 'key23']
        to_return :
            Dictionary with which configurations keys are needed for which plots
            E.g.: {'plot1': ['key9', 'key11', 'key21', 'key23']}
        '''
        to_calculate = []
        to_return = {}
        for plot_number, plot_info in requests.items():
            to_return.update({plot_number: []})
            for key, execution_entry in execution_history.items():
                if not (plot_info['is_defended'] == ('defence' in execution_entry)):
                    continue
                if MainMetricsRunner.execution_entry_matches_constants(execution_entry, plot_info['constants']):
                    to_calculate.append(key)
                    to_return[plot_number].append(key)

        return to_calculate, to_return

    @staticmethod
    def execution_entry_matches_constants(execution_entry, constants):
        '''Checks if a given configuration has matching constants
        I.e. the parameters have the values dictated by the constants.

        Parameters
        ----------
        execution_entry :
            Path that was taken to achieve the classifier, as well as any other necessary information
            For shape please refer to the documentation above at __init__.
        constants:
            Dictionary with parameters that need to have a specific value
            For shape please refer to the documentation above at __init__.
        '''
        for key, constant in constants.items():
            if not execution_entry[key] == constant['value']:
                return False
        return True

    @staticmethod
    def format_for_return(requests, to_return, metrics, debug=False):
        '''Formats metrics to be in the right shape

        Parameters
        ----------
        requests :
            Requests for different graphs specified by the user
            For shape please refer to the documentation above at __init__.
        to_return :
            Dictionary with which configurations keys are needed for which plots
            E.g.: {'plot1': ['key9', 'key11', 'key21', 'key23']}
        metrics:
            Configuration of parameters (refer to the documentation above at __init__) appended with the calculated metrics
            Difference with the shape it had at __init__, where values between <> can vary:
            'key1': {
                    ...,
                    metrics: {
                        <accuracy_on_benign>: <calculated accuracy on benign input>,
                        <accuracy_on_poison>: <calculated accuracy on poisoned input>,
                        <cad>: <difference between accuracy clean classifier and this classifier>
                    }
                }
        debug :
            Debug enables helpful print messages. Optional, default is False.

        Returns
        ----------
        Response which has the appropriate shape for further processing in the front-end
        Takes the following shape, where values between <> can vary and plotX from requests maps to metricsX:
        {
            metrics1: {
                metric: <human-readable name of the metric>,
                x_axis: <human-readable parameter along the x-axis>,
                plot: <human-readable name for plot>
                graph:
                {
                    <value1 for plot>: {
                        name: <value1 for plot>,
                        points: [
                            {x: <value for x>, y: <corresponding value for y>},
                            ...
                            {x: <value for x>, y: <corresponding value for y>}
                        ]
                    },
                    ...,
                    <valueN for plot>: {
                        name: <valueN for plot>,
                        points: [
                            {x: <value for x>, y: <corresponding value for y>},
                            ...
                            {x: <value for x>, y: <corresponding value for y>}
                        ]
                    }
                }
            },
            ...,
            metricsN: {...}
        }
        '''
        response = {}

        for plot_number, plot_info in requests.items():
            graph = {}

            for key in to_return[plot_number]:
                graph.setdefault(metrics[key][plot_info['plot']['name']], {'points': []})
                graph[metrics[key][plot_info['plot']['name']]]['points'].append({
                    'x': metrics[key][plot_info['x_axis']['name']],
                    'y': metrics[key]['metric'][plot_info['metric']['name']]})

            for key, value in graph.items():
                graph[key].update({'name': key})

            response.update({plot_number.replace('plot', 'metrics'): {
                'metric': plot_info['metric']['pretty_name'],
                'x_axis': plot_info['x_axis']['pretty_name'],
                'plot': plot_info['plot']['pretty_name'],
                'graph': graph
            }})

        if debug:
            print('Created the following format:\n{0}'.format(response))

        return response
