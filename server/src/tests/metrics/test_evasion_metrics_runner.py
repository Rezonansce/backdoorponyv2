import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from backdoorpony.metrics.evasion_metrics_runner import \
    EvasionMetricsRunner


@unittest.mock.patch('backdoorpony.metrics.abstract_metrics_runner.AbstractMetricsRunner.accuracy', return_value=(42, 3))
class TestMainMetricsRunner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy = MagicMock()

    def test_return_format(cls, calculator):
        dict_others = {
            'poison_classifier': cls.dummy,
            'poison_inputs': cls.dummy,
            'poison_labels': cls.dummy,
        }
        poisoning_m_r = EvasionMetricsRunner(80, cls.dummy, cls.dummy, dict_others)
        cls.assertEqual(poisoning_m_r.get_results(),
        {'acc': 42, 'asr': 42, 'cad': 38})

if __name__ == '__main__':
    unittest.main()
