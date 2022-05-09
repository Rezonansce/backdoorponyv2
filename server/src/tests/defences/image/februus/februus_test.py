from unittest import TestCase
from backdoorpony.datasets.CIFAR10 import CIFAR10


class TestDataLoader(TestCase):
    def initialize_februus(self):
        cifar = CIFAR10(1000)
        print("OK")