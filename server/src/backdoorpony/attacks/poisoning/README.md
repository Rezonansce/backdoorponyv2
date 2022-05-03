{::comment}
    TODO update
{:/comment}

# Poisoning attacks

Attacks of this type take clean input and add a specific trigger to them. They also create new labels for the inputs that were poisoned such that a classifier, when retrained, will have a higher likelihood of classifying an input as the target class when the input contains the trigger. Attacks of this type return a few arrays, one with mixed clean/poisoned data, one with their corresponding labels and one with the indices of the poisoned input.

Every module in this package should have a run method and some fields defined according to the following conventions.

## Inputs for run method
* params: a dict of keys and their values (as sent by the GUI, can be default values, but do no longer contain human readable strings)
* data: a numpy array with clean data
* labels: a numpy array with the clean data labels
* ran_with: a dict to keep track of the parameters and the attacks/defences run to show as a description to the metrics


## Output of the run method
* is_poison: a numpy array indicating the items in poisoned_data that are actually poisoned
* poisoned_data: a numpy array with mixed poisoned and clean data
* poisoned_labels: a numpy array with the labels for the mixed poisoned and clean data
* ran_with: a dict to keep track of the parameters and the attacks/defences run to show as a description to the metrics


## Fields
* `__name__`: the name of the attack
* `__category__`: in this folder always set to "poisoning"
* `__defaults__`: a dict with keys used by both frontend and backend with as values always an array with two items, the first one a human readable string of the parameter name (for use in the GUI) and an array of default values (can also be just one). For example [{"triggerStyle": ["Style of trigger", ['pattern', 'pixel]]}]
