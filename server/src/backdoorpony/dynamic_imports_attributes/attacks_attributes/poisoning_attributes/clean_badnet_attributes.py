__module_name_test__ = 'badnet_test'
__category_test__ = 'poisoning_test'
__input_type_test__ = 'image_test'
__defaults_test__ = {
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
    },
    'eps': {
        'pretty_name': 'Maximum perturbation',
        'default_value': [0.3],
        'info': 'Maximum perturbation that the attacker can introduce.'
    },
    'eps_step': {
        'pretty_name': 'Step size',
        'default_value': [0.1],
        'info': 'Attack step size (input variation) at each iteration.'
    },
    'max_iter': {
        'pretty_name': 'Epoch',
        'default_value': [100],
        'info': 'The maximum number of iterations'
    },
    'num_random_init': {
        'pretty_name': 'Number of random initialisations',
        'default_value': [0],
        'info': 'Number of random initialisations within the epsilon ball. For num_random_init=0 starting at the original input.'
    }
}
__link_test__ = 'https://people.csail.mit.edu/madry/lab/cleanlabel.pdf'
__info_test__ = ''' Clean Badnet is a badnet attack where the target class will be assigned using clean lable attack method. The attack will add a backdoor to a neural network by retraining the neural network on partially poisoned input.
The input is poisoned by adding a visual trigger to it. This trigger could be a pattern or just a single pixel.'''.replace('\n', '')
