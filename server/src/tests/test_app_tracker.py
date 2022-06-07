import unittest
from unittest import TestCase

from backdoorpony.app_tracker import AppTracker

class TestAppTracker(TestCase):

    def test_reset_action_info(self):
        app_tracker = AppTracker()
        app_tracker.attack_name = 'name'
        app_tracker.attack_category = 'category'
        app_tracker.attack_params = {'key': 'value'}
        app_tracker.defence_name = 'name'
        app_tracker.defence_category = 'category'
        app_tracker.defence_params = {'key': 'value'}

        app_tracker.reset_action_info()

        self.assertEqual(app_tracker.attack_name, '')
        self.assertEqual(app_tracker.attack_category, '')
        self.assertEqual(app_tracker.attack_params, {})
        self.assertEqual(app_tracker.defence_name, '')
        self.assertEqual(app_tracker.defence_category, '')
        self.assertEqual(app_tracker.defence_params, {})

    def test_generate_configuration_file_attack(self):
        app_tracker = AppTracker()
        app_tracker.dataset = 'MNIST'
        app_tracker.file_name = 'cmnist.model.pth'
        app_tracker.attack_name = 'badnet'
        app_tracker.attack_params = {'poison_percent': {'value': [0.33], 'pretty_name': 'Percentage of poison'}, 'target_class': {'value': [2], 'pretty_name': 'Target class'}, 'trigger_style': {'value': ['pattern'], 'pretty_name': 'Style of trigger'}}
        self.assertEqual(app_tracker.generate_configuration_file(), [
            "1. Select the MNIST dataset;",
            "2. Upload your initial model named 'cmnist.model.pth';",
            "3. Choose the badnet attack;",
            "    3a. Input [0.33] for percentage of poison;",
            "    3b. Input [2] for target class;",
            "    3c. Input ['pattern'] for style of trigger;",
            "4. Select 'no defence';",
            "5. Press execute."
        ])

    def test_generate_configuration_file_defence(self):
        app_tracker = AppTracker()
        app_tracker.dataset = 'MNIST'
        app_tracker.defence_name = 'strip'
        app_tracker.defence_params = {'number_of_images': {'pretty_name': 'Number of images','value': [100]}}
        self.assertEqual(app_tracker.generate_configuration_file(), [
            "1. Select the MNIST dataset;",
            "2. Select \'use built-in model\';",
            "3. Select 'no attack';",
            "4. Choose the strip defence;",
            "    4a. Input [100] for number of images;",
            "5. Press execute."
        ])

if __name__ == '__main__':
    unittest.main()
    
