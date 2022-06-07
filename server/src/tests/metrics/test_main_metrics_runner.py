import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from backdoorpony.metrics.main_metrics_runner import MainMetricsRunner


@unittest.mock.patch('backdoorpony.metrics.abstract_metrics_runner.AbstractMetricsRunner.accuracy', return_value=(42, 3))
@unittest.mock.patch('backdoorpony.metrics.poisoning_metrics_runner.PoisoningMetricsRunner.__init__', return_value=None)
@unittest.mock.patch('backdoorpony.metrics.poisoning_metrics_runner.PoisoningMetricsRunner.get_results', return_value={'acc': 42})
class TestMainMetricsRunner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy = MagicMock()
        cls.execution_history_1 = {
            'key1': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy
            },
            'key2': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy
            },
            'key3': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy
            },
            'key4': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key5': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pixel', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key6': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pixel', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key7': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pixel', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key8': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pixel', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key9': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pixel', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key10': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pixel', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key11': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pixel', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key12': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pixel', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key13': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key14': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key15': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key16': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key17': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pattern', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key18': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pattern', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key19': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pattern', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key20': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .5, 'attackStyle': 'pattern', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key21': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pattern', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key22': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pattern', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy,
            },
            'key23': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pattern', 'param': True, 'otherParam': 5, 'dict_others': cls.dummy,
            },
            'key24': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .75, 'attackStyle': 'pattern', 'param': False, 'otherParam': 5, 'dict_others': cls.dummy
            }
        }
        cls.execution_history_2 = {
            'key1': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'dict_others': cls.dummy
            },
            'key2': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'dict_others': cls.dummy
            },
            'key3': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'defence': 'strip', 'defenceCategory': 'transformer', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy
            },
            'key4': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pixel', 'defence': 'strip', 'defenceCategory': 'transformer', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy
            },
            'key5': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'defence': 'strip', 'defenceCategory': 'transformer', 'param': True, 'otherParam': 1, 'dict_others': cls.dummy
            },
            'key6': {
                'attack': 'badnet', 'attackCategory': 'poisoning', 'poisonPercent': .33, 'attackStyle': 'pattern', 'defence': 'strip', 'defenceCategory': 'transformer', 'param': False, 'otherParam': 1, 'dict_others': cls.dummy
            }
        }
        cls.request_1 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'pretty_name': 'Attack style',
                    'name': 'attackStyle'
                },
                'x_axis': {
                    'pretty_name': 'Poison percent',
                    'name': 'poisonPercent'
                },
                'is_defended': False,
                'constants': {}
            }
        }
        cls.request_2 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'pretty_name': 'Attack style',
                    'name': 'attackStyle'
                },
                'x_axis': {
                    'pretty_name': 'Poison percent',
                    'name': 'poisonPercent'
                },
                'is_defended': False,
                'constants': {
                    'param': {
                        'pretty_name': 'parameter',
                        'value': True
                    },
                    'otherParam': {
                        'pretty_name': 'other parameter',
                        'value': 1
                    }
                }
            }
        }
        cls.request_3 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'pretty_name': 'Attack style',
                    'name': 'attackStyle'
                },
                'x_axis': {
                    'pretty_name': 'Other parameter',
                    'name': 'otherParam'
                },
                'is_defended': False,
                'constants': {
                    'poisonPercent': {
                        'pretty_name': 'Poison percent',
                        'value': .75
                    },
                    'param': {
                        'pretty_name': 'parameter',
                        'value': True
                    }
                }
            }
        }
        cls.request_4 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'pretty_name': 'Other parameter',
                    'name': 'otherParam'
                },
                'x_axis': {
                    'pretty_name': 'Parameter',
                    'name': 'param'
                },
                'is_defended': False,
                'constants': {
                    'attackStyle': {
                        'pretty_name': 'attack style',
                        'value': 'pattern'
                    }
                }
            }
        }
        cls.request_5 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'pretty_name': 'Parameter',
                    'name': 'param'
                },
                'x_axis': {
                    'pretty_name': 'Attack style',
                    'name': 'attackStyle'
                },
                'is_defended': True,
                'constants': {
                    'poisonPercent': {
                        'pretty_name': 'Poison percent',
                        'value': .33
                    },
                    'otherParam': {
                        'pretty_name': 'other parameter',
                        'value': 1
                    }
                }
            }
        }
        cls.request_6 = {
            'plot1': {
                'metric': {
                    'pretty_name': 'Accuracy',
                    'name': 'acc'
                },
                'plot': {
                    'poisonPercent': {
                        'pretty_name': 'Poison percent',
                        'name': 'poisonPercent'
                    }
                },
                'x_axis': {
                    'pretty_name': 'Attack style',
                    'name': 'attackStyle'
                },
                'is_defended': False,
                'constants': {
                }
            }
        }

    def test_filter_empty(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_1, cls.request_1)
        cls.assertEqual(len(to_calculate), 24)

    def test_filter_integer_boolean(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_1, cls.request_2)
        cls.assertEqual(len(to_calculate), 6)

    def test_filter_float_boolean(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_1, cls.request_3)
        cls.assertEqual(len(to_calculate), 4)

    def test_filter_string(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_1, cls.request_4)
        cls.assertEqual(len(to_calculate), 12)

    def test_filter_defence_true(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_2, cls.request_5)
        cls.assertEqual(len(to_calculate), 4)

    def test_filter_defence_false(cls, calculator, init_runner, results_runner):
        to_calculate, to_return = MainMetricsRunner.filter_execution_history(cls.execution_history_2, cls.request_6)
        cls.assertEqual(len(to_calculate), 2)

    def test_format(cls, calculator, init_runner, results_runner):
        main_m_r = MainMetricsRunner()
        main_m_r.instantiate(cls.dummy, cls.execution_history_1, (cls.dummy, cls.dummy), cls.request_2)
        cls.assertEqual(main_m_r.get_results(), {
            'metrics1': {
                'metric': 'Accuracy',
                'x_axis': 'Poison percent',
                'plot': 'Attack style',
                'graph': {
                    'pixel': {
                        'name': 'pixel',
                        'points': [
                            {'x': 0.33, 'y': 42},
                            {'x': 0.5, 'y': 42},
                            {'x': 0.75, 'y': 42}
                        ]
                    },
                    'pattern': {
                        'name': 'pattern',
                        'points': [
                            {'x': 0.33, 'y': 42},
                            {'x': 0.5, 'y': 42},
                            {'x': 0.75, 'y': 42}
                        ]
                    }
                }
            }
        })

    def test_update_function_equal(cls, calculator, init_runner, results_runner):
        main_m_r = MainMetricsRunner()
        main_m_r.instantiate(cls.dummy, cls.execution_history_1, (cls.dummy, cls.dummy), cls.request_2)
        cls.assertEqual(6, calculator.call_count)
        main_m_r.update(cls.request_2)
        cls.assertEqual(6, calculator.call_count)
        # call count should not increase as all these configurations have already been calculated

    def test_update_function_more(cls, calculator, init_runner, results_runner):
        main_m_r = MainMetricsRunner()
        main_m_r.instantiate(cls.dummy, cls.execution_history_1, (cls.dummy, cls.dummy), cls.request_2)
        cls.assertEqual(6, calculator.call_count)
        main_m_r.update(cls.request_1)
        cls.assertEqual(24, calculator.call_count)
        # call count should increase by 21 as there are only 21 new configurations

    def test_update_function_less(cls, calculator, init_runner, results_runner):
        main_m_r = MainMetricsRunner()
        main_m_r.instantiate(cls.dummy, cls.execution_history_1, (cls.dummy, cls.dummy), cls.request_1)
        cls.assertEqual(24, calculator.call_count)
        main_m_r.update(cls.request_2)
        cls.assertEqual(24, calculator.call_count)
        # call count should not increase as all these configurations have already been calculated
        cls.assertEqual(main_m_r.get_results(), {
            'metrics1': {
                'metric': 'Accuracy',
                'x_axis': 'Poison percent',
                'plot': 'Attack style',
                'graph': {
                    'pixel': {
                        'name': 'pixel',
                        'points': [
                            {'x': 0.33, 'y': 42},
                            {'x': 0.5, 'y': 42},
                            {'x': 0.75, 'y': 42}
                        ]
                    },
                    'pattern': {
                        'name': 'pattern',
                        'points': [
                            {'x': 0.33, 'y': 42},
                            {'x': 0.5, 'y': 42},
                            {'x': 0.75, 'y': 42}
                        ]
                    }
                }
            }
        })

if __name__ == '__main__':
    unittest.main()
