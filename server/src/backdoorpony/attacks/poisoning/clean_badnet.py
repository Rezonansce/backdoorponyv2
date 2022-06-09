'''
Run the BadNet attack to generate training data that contains a trigger.
For documentation check the README inside the attacks/poisoning folder.
'''
from copy import deepcopy

import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.clean_label_backdoor_attack import PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import (add_pattern_bd, add_single_bd)

__name__ = 'clean_badnet'
__category__ = 'poisoning'
__input_type__ = 'image'
__defaults__ = {
    'trigger_style': {
        'pretty_name': 'Style of trigger',
        'default_value': ['pattern'],
        'info': 'The trigger style, as the name suggests, determines the style of the trigger that is applied to the images. The style could either be \'pixel\' or \'pattern\'. The pixel is almost invisible to humans, but its subtlety negatively affects the effectiveness. The pattern is a reverse lambda that is clearly visible for humans, but it is also more effective.'
    },
    'poison_percent': {
        'pretty_name': 'Percentage of poison',
        'default_value':  [0.1],
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
__link__ = 'https://people.csail.mit.edu/madry/lab/cleanlabel.pdf'
__info__ = '''Badnet is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input.
The input is poisoned by adding a visual trigger to it. This trigger could be a pattern or just a single pixel. The target class will be assigned using clean lable attack method'''.replace('\n', '')


def run(clean_classifier, train_data, test_data, execution_history, attack_params):
    '''Runs the Clean label badnet attack

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    train_data :
        Data that the clean classifier was trained on as a tuple with (inputs, labels)
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    attack_params :
        Dictionary with the parameters for the attack (one value per parameter)

    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating a BadNet attack.')
    key_index = 0
    train_images = train_data[0]
    train_labels = train_data[1]
    test_images = test_data[0]
    test_labels = test_data[1]

    for ts in range(len(attack_params['trigger_style']['value'])):
        for tc in range(len(attack_params['target_class']['value'])):
            for iter_eps in range(len(attack_params['eps']['value'])):
                for iter_eps_step in range(len(attack_params['eps_step']['value'])):
                    for iter_max_iter in range(len(attack_params['max_iter']['value'])):
                        for iter_num_random_init in range(len(attack_params['num_random_init']['value'])):
                            _, full_poison_data, full_poison_labels = CleanBadNet(attack_params['trigger_style']['value'][ts],
                                                                            0.99,
                                                                            attack_params['target_class']['value'][tc],
                                                                            clean_classifier,
                                                                            attack_params['eps']['value'][iter_eps],
                                                                            attack_params['eps_step']['value'][iter_eps_step],
                                                                            attack_params['max_iter']['value'][iter_max_iter],
                                                                            attack_params['num_random_init']['value'][iter_num_random_init])\
                                                                            .poison(deepcopy(train_images), deepcopy(train_labels), True)

                            for pp in range(len(attack_params['poison_percent']['value'])):
                                # Run the attack for a combination of input variales
                                execution_entry = {}
                                _, poisoned_train_data, poisoned_train_labels = CleanBadNet(
                                    attack_params['trigger_style']['value'][ts],
                                    attack_params['poison_percent']['value'][pp],
                                    attack_params['target_class']['value'][tc],
                                    clean_classifier,
                                    attack_params['eps']['value'][iter_eps],
                                    attack_params['eps_step']['value'][iter_eps_step],
                                    attack_params['max_iter']['value'][iter_max_iter],
                                    attack_params['num_random_init']['value'][iter_num_random_init]).poison(deepcopy(train_images), deepcopy(train_labels), True)
                                is_poison_test, poisoned_test_data, poisoned_test_labels = CleanBadNet(
                                    attack_params['trigger_style']['value'][ts],
                                    attack_params['poison_percent']['value'][pp],
                                    attack_params['target_class']['value'][tc],
                                    clean_classifier,
                                    attack_params['eps']['value'][iter_eps],
                                    attack_params['eps_step']['value'][iter_eps_step],
                                    attack_params['max_iter']['value'][iter_max_iter],
                                    attack_params['num_random_init']['value'][iter_num_random_init]).poison(deepcopy(test_images), deepcopy(test_labels), False)

                                poisoned_classifier = deepcopy(clean_classifier)
                                poisoned_classifier.fit(poisoned_train_data, poisoned_train_labels)

                                execution_entry.update({
                                    'attack': __name__,
                                    'attackCategory': __category__,
                                    'trigger_style': attack_params['trigger_style']['value'][ts],
                                    'poison_percent': attack_params['poison_percent']['value'][pp],
                                    'target_class': attack_params['target_class']['value'][tc],
                                    'dict_others': {
                                        'poison_classifier': deepcopy(poisoned_classifier),
                                        'poison_inputs': deepcopy(full_poison_data),
                                        'poison_labels': deepcopy(full_poison_labels),
                                        'is_poison_test': deepcopy(is_poison_test),
                                        'poisoned_test_data': deepcopy(poisoned_test_data),
                                        'poisoned_test_labels': deepcopy(poisoned_test_labels)
                                    }
                                })

                                key_index += 1
                                execution_history.update({'badnet' + str(key_index): execution_entry})

    return execution_history


class CleanBadNet(object):
    def __init__(self, modification_type, percent_poison, target_class, proxy_classifier, eps, eps_step, max_iter, num_random_init, path='../assets/alert.png'):
        self.modification_type = modification_type
        self.path = path
        self.percent_poison = percent_poison
        self.target_class = target_class
        self.proxy_classifier = proxy_classifier
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.num_random_init = num_random_init

    def add_modification(self, x):
        '''
        Determine the type of trigger to poison the data with.
        Pick from pattern, pixel and image.
        When image is specified the path to an image to use as backdoor trigger can also be specified.
        Uses ART libraries.
        '''
        if self.modification_type == 'pattern':
            # add the pattern on the correct axes then reformat
            x = np.swapaxes(x, 1, 3)
            x = add_pattern_bd(x, pixel_value=self.max_val)
            x = np.swapaxes(x, 1, 3)
            return x
        elif self.modification_type == 'pixel':
            return add_single_bd(x, pixel_value=self.max_val)
        elif self.modification_type == 'image':
            raise('Currently broken')
            # return insert_image(x, backdoor_path=self.path, size=(10, 10), channels_first=True)
        else:
            raise('Unknown backdoor type')

    def poison(self, input_data, input_labels, shuffle):
        '''
        Add the trigger to a certain percentage of the data.
        Specify a target class for the poisoned data to be evaluated to.

        Parameters
        ----------
        x_clean :
            The dataset to poison
        y_clean :
            Labels corresponding to x_clean
        percent_poison :
            A float between 0 and 1 representing the percentage of the data to poison
        target_class :
            The index of the target class for the backdoor to use
        shuffle :
            Boolean value used to determine whether the dataset should be shuffled
            Recommended for training data, not for test data.

        Returns
        ----------
        is_poison :
            A boolean array of the same size as the training data to indicate what indices have been poisoned.
        x_poison :
            The (partially) poisoned data.
        y_poison :
            The labels corresponding to the (partially) poisoned data.
        '''

        # Save the maximum of the training data for the actual poisoning
        self.max_val = np.max(input_data)
        x_to_poison = np.copy(input_data)
        y_to_poison = np.copy(input_labels)

        # Get the unique classes in input_labels
        classes = np.unique(input_labels)

        # Remove the target class from the unique classes
        classes = np.delete(classes, self.target_class)

        #  Save the target classes and remove them from to_poison
        target_class_x = x_to_poison[y_to_poison == self.target_class]
        target_class_y = y_to_poison[y_to_poison == self.target_class]
        x_to_poison = x_to_poison[y_to_poison != self.target_class]
        y_to_poison = y_to_poison[y_to_poison != self.target_class]

        # produce empty arrays for new data with correct (modular) shapes
        shape_x = list(np.shape(x_to_poison))
        shape_x[0] = 0
        shape_x = tuple(shape_x)

        x_poison = np.empty(shape_x)
        y_poison = np.empty((0, 1))

        x_clean = np.empty(shape_x)
        y_clean = np.empty((0, 1))

        for current_class in classes:
            # get current class features
            x_current_class = x_to_poison[y_to_poison == current_class]

            # calculate the number of data entries to poison
            num_imgs_to_poison = round(
                (self.percent_poison * len(x_current_class)))

            # Select poison data,
            # do not allow poisoning twice (replace=False)
            indices_to_poison = np.random.choice(
                len(x_current_class), num_imgs_to_poison, replace=False)

            # Split in poison and clean
            x_poison_current_class = x_current_class[indices_to_poison]

            # remove to_poison data entries to get the clean ones
            clean_indices = np.delete(
                np.arange(len(x_current_class)), indices_to_poison)
            x_clean_current_class = x_current_class[clean_indices]

            # get unpoisoned data in the final dataset
            x_clean = np.append(x_clean, x_clean_current_class, axis=0)

            # make sure shape is 4d, (num_entries, num_channels, width, height)
            y_clean = np.append(
                y_clean, np.zeros((len(x_clean_current_class), 1)) * current_class , axis=0)

            # Actually poison the poison partition
            if (num_imgs_to_poison > 0):
                backdoor_attack = PoisoningAttackBackdoor(
                    self.add_modification)

                # create an attack using art framework
                clean_backdoor_attack = PoisoningAttackCleanLabelBackdoor(
                    backdoor=backdoor_attack, proxy_classifier=self.proxy_classifier, target=np.ones((num_imgs_to_poison, 1)) * self.target_class,
                    pp_poison=self.percent_poison, eps=self.eps, eps_step=self.eps_step, max_iter=self.max_iter,
                    num_random_init=self.num_random_init)

                # utilize the attack to poison data
                x_poison_current_class, poison_labels = clean_backdoor_attack.poison(
                    x=x_poison_current_class, y=np.ones((num_imgs_to_poison, 1)) * self.target_class, broadcast=False)


                # Append poisoned data to the poison class
                x_poison = np.append(x_poison, x_poison_current_class, axis=0)
                y_poison = np.append(y_poison, poison_labels, axis=0)

        # Create new arrays for final data (modular shape)
        is_poison = np.empty((0,))
        x_combined = np.empty(shape_x)
        y_combined = np.empty((0, ))

        # Add the items which originally had the target_class to the combined set
        x_combined = np.append(x_combined, target_class_x, axis=0)
        y_combined = np.append(y_combined, target_class_y, axis=0)

        # Add the items which are unpoisoned to the combined set
        x_combined = np.append(x_combined, x_clean, axis=0)

        # reshape y_clean from (num_entries, 1) into (num_entries, ) before combining
        # TODO remove squeeze when adding support for more than 1 channel, metrics runners also need to be updated accordingly
        y_combined = np.append(y_combined, np.squeeze(y_clean), axis=0)

        # Mark the unpoisoned data and the data that was
        # originally the target_class as unpoisoned
        is_poison = np.append(is_poison, np.zeros(len(y_combined)))

        # Add the items which are poisoned to the combined set
        x_combined = np.append(x_combined, x_poison, axis=0)

        # reshape y_poison from (num_entries, 1) into (num_entries, ) before combining
        # TODO remove squeeze when adding support for more than 1 channel, metrics runners also need to be updated accordingly
        y_combined = np.append(y_combined, np.squeeze(y_poison), axis=0)

        # Mark poisoned data as such
        is_poison = np.append(is_poison, np.ones(len(np.squeeze(y_poison))))

        # Convert to a boolean array
        is_poison = is_poison != 0

        # Shuffle data for a more even spread (only recommended for train data).
        if shuffle:
            n_train = np.shape(y_combined)[0]
            shuffled_indices = np.arange(n_train)
            np.random.shuffle(shuffled_indices)
            x_combined = x_combined[shuffled_indices]
            y_combined = y_combined[shuffled_indices]

        # reshape x_combined from (num_entries, 1, width, height) into (num_entries, widt, height) before combining
        # this is required because metrics runners don't currently work with channels
        # TODO remove squeeze when adding support for more than 1 channel, metrics runners also need to be updated accordingly
        return is_poison, np.squeeze(x_combined), y_combined
