import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import backdoorpony.defences.transformer.poisoning.strip
import numpy as np
from backdoorpony.defences.transformer.poisoning.strip import STRIP
from numpy.testing._private.utils import assert_equal


class TestStrip(TestCase):
    # def test_run(self):
    #     # Test the run method of STRIP
    #     # Arange
    #     # Make sure we can control the init method of the class we call.
    #     with patch('backdoorpony.defences.transformer.poisoning.strip.STRIP.__init__') as patch_strip:
    #         patch_strip.return_value = None
    #         mock_classifier = MagicMock(name='classifier')

    #         # Act
    #         defence_classifier, poison_condition, ran_with = backdoorpony.defences.transformer.poisoning.strip.run(
    #             {"numberOfImages": [100, 150]}, np.array([1, 2, 3]), mock_classifier, dict())

    #         # Assert
    #         # Check if the defence was run with the correct params.
    #         assert_equal(ran_with['numberOfImages'], 100)
    #         assert_equal(ran_with['defence'], "strip")
    #         assert_equal(ran_with['defenceCategory'], "transformer")
    #         # Check if the lambda function that we want to get returned
    #         # is indeed the one that is returned, this check uses bytecode
    #         assert_equal(poison_condition.__code__.co_code, (lambda x: (
    #             x == np.zeros(len(x))).all()).__code__.co_code)
    #         # Assert that the defence classifier is an instantiation of STRIP
    #         self.assertTrue(isinstance(defence_classifier, STRIP))

    def test_init(self):
        # Test __init__ of STRIP
        # Arange
        # Make sure to not actually call ART
        with patch('art.defences.transformer.poisoning.STRIP.__init__') as patch_strip_init:
            with patch('art.defences.transformer.poisoning.STRIP.__call__') as patch_strip_call:
                mock_classifier = MagicMock(name='classifier')
                defenceImages = 2
                clean = np.array([1, 2, 3])
                # Make sure the data returned by the patched calls to ART is usable
                patch_strip_init.return_value = None
                patch_strip_call.return_value = mock_classifier

                # Act
                STRIP(mock_classifier, clean, defenceImages)

                # Assert
                # Make sure the init function correctly calls the
                # (patched) init functions from ART
                patch_strip_init.assert_called_once()
                patch_strip_call.assert_called_once()
                mock_classifier.mitigate.assert_called_once()

    def test_get_predictions(self):
        # Test get_predictions of STRIP
        # Arange
        # First we mock the classifier again and prevent actually calling ART libraries.
        with patch('art.defences.transformer.poisoning.STRIP.__init__') as patch_strip_init:
            with patch('art.defences.transformer.poisoning.STRIP.__call__') as patch_strip_call:
                mock_classifier = MagicMock(name='classifier')
                defenceImages = 2
                clean = np.array([1, 2, 3])
                patch_strip_init.return_value = None
                patch_strip_call.return_value = mock_classifier
                test = np.array([4, 5, 6])

                # Here we mock the return value of the predict
                # method in the mocked classifier
                # based on the input it is given.
                def side_effect_func(value):
                    if np.array_equal(value, test):
                        return np.array([1, 1, 1])
                    elif np.array_equal(value, clean[defenceImages:]):
                        return np.array([0, 0, 0])
                    else:
                        return np.array([-1, -1, -1])

                mock_classifier.predict = MagicMock(
                    side_effect=side_effect_func)
                defence = STRIP(mock_classifier, clean, defenceImages)

                # Act
                poison_preds, clean_preds = defence.get_predictions(test)

                # Assert
                # Make sure the predict function was called two
                # times with different inputs
                calls = [call(test), call(clean[defenceImages:])]
                mock_classifier.predict.assert_has_calls(calls)
                assert_equal(poison_preds, np.array([1, 1, 1]))
                assert_equal(clean_preds, np.array([0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
