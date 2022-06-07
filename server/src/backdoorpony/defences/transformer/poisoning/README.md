{::comment}
    TODO update
{:/comment}

# Transformer defences against poisoning attacks

Defences of this type try to modify a poisoned classifier by superimposing on clean data and finding the poison trigger. They return a cleaned (to the best of the defence's ability) classifier.

Every module in this package should have a run method and some fields defined according to the following conventions.

## Inputs for run method
* params: a dict of keys and their values (as sent by the GUI)
* data: a numpy array with clean data
* ran_with: a string to keep track of the parameters and the attacks/defences run to show as a description to the metrics

## Output of the run method
* classifier: a cleaned classifier
* ran_with: a string to keep track of the parameters and the attacks/defences run to show as a description to the metrics

## Fields
* `__name__`: the name of the defence
* `__category__`: in this folder always set to "transformer"
* `__defaults__`: a dict with keys used by both frontend and backend with as values always an array with two items, the first one a human readable string of the parameter name (for use in the GUI) and an array of default values (can also be just one)
