import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from backdoorpony.metrics.transformer_metrics_runner import \
    TransformerMetricsRunner


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
            'poison_condition': cls.dummy
        }
        transformer_m_r = TransformerMetricsRunner(80, cls.dummy, cls.dummy, dict_others)
        cls.assertEqual(transformer_m_r.get_results(),
        {'acc': 42, 'asr': 42, 'cad': 38, 'tpr': 3, 'fpr': 3, 'tnr': 97, 'fnr': 97})

if __name__ == '__main__':
    unittest.main()
