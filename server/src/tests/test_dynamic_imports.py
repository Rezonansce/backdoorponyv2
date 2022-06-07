from importlib import import_module
from types import ModuleType
import unittest
from unittest import TestCase
from backdoorpony.dynamic_imports import import_submodules_attributes, get_as_package
import backdoorpony.dynamic_imports_attributes


class TestDynamicImports(TestCase):

    def test_import_badnet_test_defaults_attributes(self):
        _, attributes_modules = import_submodules_attributes(package=backdoorpony.dynamic_imports_attributes, req_attr=['__defaults_test__'], result=[], req_module='badnet_attributes')
        attributes_modules_expected = {
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
        self.assertEqual(attributes_modules[0]['defaults_test'], attributes_modules_expected)

    def test_import_badnet_defaults_types(self):
        imports, _ = import_submodules_attributes(package=backdoorpony.dynamic_imports_attributes, req_attr=[
            '__class_name_test__', '__defaults_test__'], result=[], req_module='badnet_attributes')
        poisoning = 'backdoorpony.dynamic_imports_attributes.attacks_attributes.poisoning_attributes'
        badnet = 'backdoorpony.dynamic_imports_attributes.attacks_attributes.poisoning_attributes.badnet_attributes'
        self.assertEqual(type(imports[poisoning]), ModuleType)
        self.assertEqual(type(imports[badnet]), ModuleType)
        self.assertEqual(imports[poisoning].__name__, poisoning)

    def test_import_metrics_info_attributes(self):
        _, attributes_modules = import_submodules_attributes(package=backdoorpony.dynamic_imports_attributes, result=[], req_attr=[
            '__category_test__', '__info_test__'])
        evasion_info_expected = {'acc': {'pretty_name': 'Accuracy', 'info': 'The accuracy is the probability that the classifier predicts the correct class for any given input. This metric is calculated using clean inputs only and is reported in percentages. The higher the accuracy, the better the classifier is at making correct predictions.'}, 'asr': {'pretty_name': 'ASR', 'info': 'ASR, or Attack Success Rate, is the probability that the classifier predicts a given poisoned input as belonging to the target class. This metric is calculated by removing inputs where the source class and target class are the same and poisoning the remainder. ASR is reported in percentages. The higher it is, the larger the chance the classifier will predict a poisoned input to belong to the target class.'}, 'cad': {
            'pretty_name': 'CAD', 'info': 'CAD, or Clean Accuracy Drop, is the difference in accuracy between the clean classifier and the modified classifier on benign input. The metric is calculated by subtracting the accuracy of the modified classifier from that of the clean classifier. This subtraction results in a percentage that can be positive or negative. A positive CAD means that the clean classifier was more accurate than the modified classifier, while a negative CAD means the modifier was the more accurate of the two.'}}
        poisoning_info_expected = {'acc': {'pretty_name': 'Accuracy', 'info': 'The accuracy is the probability, in percentages, that the classifier predicts the correct class for any given input. This metric is calculated using clean inputs only.'}, 'asr': {'pretty_name': 'ASR', 'info': 'ASR, or Attack Success Rate, is the probability that the classifier predicts a given poisoned input as belonging to the target class. This metric is calculated by removing inputs where the source class and target class are the same and poisoning the remainder. ASR is reported in percentages. The higher it is, the larger the chance the classifier will predict a poisoned input to belong to the target class.'}, 'cad': {
            'pretty_name': 'CAD', 'info': 'CAD, or Clean Accuracy Drop, is the difference in accuracy between the clean classifier and the modified classifier on benign input. The metric is calculated by subtracting the accuracy of the modified classifier from that of the clean classifier. This subtraction results in a percentage that can be positive or negative. A positive CAD means that the clean classifier was more accurate than the modified classifier, while a negative CAD means the modifier was the more accurate of the two.'}}
        transformer_info_expected = {'acc': {'pretty_name': 'Accuracy', 'info': 'The accuracy is the probability, in percentages, that the classifier predicts the correct class for any given input. This metric is calculated using clean inputs only.'}, 'asr': {'pretty_name': 'ASR', 'info': 'ASR, or Attack Success Rate, is the probability that the classifier predicts a given poisoned input as belonging to the target class. This metric is calculated by removing inputs where the source class and target class are the same and poisoning the remainder. ASR is reported in percentages. The higher it is, the larger the chance the classifier will predict a poisoned input to belong to the target class.'}, 'cad': {'pretty_name': 'CAD', 'info': 'CAD, or Clean Accuracy Drop, is the difference in accuracy between the clean classifier and the modified classifier on benign input. The metric is calculated by subtracting the accuracy of the modified classifier from that of the clean classifier. This subtraction results in a percentage that can be positive or negative. A positive CAD means that the clean classifier was more accurate than the modified classifier, while a negative CAD means the modifier was the more accurate of the two.'}, 'tpr': {
            'pretty_name': 'TPR', 'info': 'TPR, or True Positive Rate, is the probability that the classifier correctly identifies a given input as poisoned. The metric is calculated by poisoning 100% of the inputs and monitoring how many of these it identifies as poisoned. The result is expressed in percentages.'}, 'fpr': {'pretty_name': 'FPR', 'info': 'FPR, or False Positive Rate, is the probability that the classifier incorrectly identifies a given input as poisoned. The metric is calculated by providing benign inputs and monitoring how many of these it identifies as poisoned. The result is expressed in percentages.'}, 'tnr': {'pretty_name': 'TNR', 'info': 'TNR, or True Negative Rate, is the probability that the classifier correctly identifies a given input as benign. The metric is calculated by providing benign inputs and monitoring how many of these it identifies as clean. The result is expressed in percentages.'}, 'fnr': {'pretty_name': 'FNR', 'info': 'FNR, or False Negative Rate, is the probability that the classifier incorrectly identifies a given input as benign. The metric is calculated by poisoning 100% of the inputs and monitoring how many of these it identifies as clean. The result is expressed in percentages.'}}
        for attributes_module in attributes_modules:
            if attributes_module['category_test'] == 'evasion':
                evasion_info = attributes_module['info_test']
            if attributes_module['category_test'] == 'transformer':
                transformer_info = attributes_module['info_test']
            if attributes_module['category_test'] == 'poisoning':
                poisoning_info = attributes_module['info_test']
        self.assertEqual(poisoning_info, poisoning_info_expected)
        self.assertEqual(transformer_info, transformer_info_expected)
        self.assertEqual(evasion_info, evasion_info_expected)

    def test_import_metrics_info_types(self):
        imports, _ = import_submodules_attributes(package=backdoorpony.dynamic_imports_attributes.metrics_attributes, req_attr=[
            '__category_test__', '__info_test__'], result=[])
        evasion = 'backdoorpony.dynamic_imports_attributes.metrics_attributes.metrics_evasion_attributes'
        poisoning = 'backdoorpony.dynamic_imports_attributes.metrics_attributes.metrics_poisoning_attributes'
        transformer = 'backdoorpony.dynamic_imports_attributes.metrics_attributes.metrics_transformer_attributes'
        self.assertEqual(type(imports[evasion]), ModuleType)
        self.assertEqual(type(imports[poisoning]), ModuleType)
        self.assertEqual(type(imports[transformer]), ModuleType)
        self.assertEqual(imports[poisoning].__module_name_test__, 'metrics')
    

    def test_get_as_package(self):
        package = get_as_package('backdoorpony.dynamic_imports_attributes.attacks_attributes.poisoning_attributes')
        print(package)
        package_name_expected = 'backdoorpony.dynamic_imports_attributes.attacks_attributes.poisoning_attributes'
        self.assertEqual(package.__name__, package_name_expected)
        self.assertEqual(type(package), ModuleType)

if __name__ == '__main__':
    unittest.main()
