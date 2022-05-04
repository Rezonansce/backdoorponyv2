import unittest
from unittest import TestCase

import numpy as np
from backdoorpony.datasets.CIFAR10 import CIFAR10

class TestDataLoader(TestCase):
    def test_get_data(self):
        cifar = CIFAR10(1)
        dataset = cifar.get_datasets()
        print(dataset)