import pkgutil

import backdoorpony


class Runner(object):
    def run_attack(self, clean_classifier, train_data, test_data, execution_history, attack_to_run, attack_params):
        '''Runs the specified attack

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
        attack_to_run :
            Name of the attack that should be executed
        attack_params :
            Dictionary with the parameters for the attack (a list of values per parameter)

        Returns
        ----------
        Returns the updated execution history dictionary
        '''
        found_attack = False

        packages = [backdoorpony.attacks.poisoning]

        for package in packages:
            prefix = package.__name__ + '.'
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
                module = __import__(modname, fromlist='dummy')

                if module.__name__ == attack_to_run:
                    found_attack = True
                    execution_history = module.run(clean_classifier, train_data, test_data,
                                                   execution_history, attack_params)

        if not found_attack:
            raise FileNotFoundError('Attack is not defined')

        return execution_history

    def run_defence(self, clean_classifier, test_data, execution_history, defence_to_run, defence_params):
        '''Runs the specified defence

        Parameters
        ----------
        clean_classifier :
            Classifier that has not been tampered with, i.e. is clean
        test_data :
            Data that the clean classifier will be validated on as a tuple with (inputs, labels)
        execution_history :
            Dictionary with paths of attacks/defences taken to achieve classifiers, if any
        defence_to_run :
            Name of the defence that should be executed
        defence_params :
            Dictionary with the parameters for the defence (a list of values per parameter)

        Returns
        ----------
        Returns the updated execution history dictionary
        '''
        found_defence = False

        # TODO if ran_with is {}, use clean_classifier (pass classifier and test_data)
        # Defence
        packages = [backdoorpony.defences.transformer.poisoning]
        for package in packages:
            prefix = package.__name__ + '.'
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
                module = __import__(modname, fromlist='dummy')

                if module.__name__ == defence_to_run:
                    found_defence = True
                    execution_history = module.run(clean_classifier, test_data,
                                                   execution_history, defence_params)

        if not found_defence:
            raise FileNotFoundError('Defence is not defined')

        return execution_history
