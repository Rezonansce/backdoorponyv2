import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from backdoorpony.metrics.abstract_metrics_runner import AbstractMetricsRunner


class TestAbstractMetricsRunner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cls.b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        cls.c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cls.dummy = MagicMock()
        cls.mocked_nn = MagicMock()

    def test_0_accuray(cls):
        preds = np.array([cls.a, cls.a, cls.a, cls.a])
        targets = np.array([8, 9, 3, 5])

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets), (0, 0))

    def test_50_accuray(cls):
        preds = np.array([cls.a, cls.a, cls.a, cls.a])
        targets = np.array([0, 5, 8, 0])

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets), (50, 0))

    def test_100_accuray(cls):
        preds = np.array([cls.a, cls.a, cls.a, cls.a])
        targets = np.array([0, 0, 0, 0])

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets), (100, 0))

    def test_50_poisoned_zeros(cls):
        preds = np.array([cls.c, cls.b, cls.b, cls.c])
        targets = np.array([7, 9, 9, 5])
        poison_condition = lambda x: (x==np.zeros(len(x))).all()

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets, poison_condition), (50, 50))

    def test_100_poisoned_zeros(cls):
        preds = np.array([cls.c, cls.c, cls.c, cls.c])
        targets = np.array([7, 2, 0, 5])
        poison_condition = lambda x: (x==np.zeros(len(x))).all()

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets, poison_condition), (0, 100))

    def test_100_poisoned_minus_one(cls):
        preds = np.array([[-1], [-1], [-1], [-1]])
        targets = np.array([7, 2, 0, 5])
        poison_condition = lambda x: x==-1

        cls.mocked_nn.predict.return_value = preds
        cls.assertEqual(AbstractMetricsRunner.accuracy(
            cls.mocked_nn, cls.dummy, targets, poison_condition), (0, 100))

if __name__ == '__main__':
    unittest.main()
