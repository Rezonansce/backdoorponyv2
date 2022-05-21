import unittest
import networkx as nx
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from backdoorpony.datasets.MUTAG import MUTAG
from backdoorpony.datasets.utils import graphbackdoor

class TestDataLoader(TestCase):
    def test_get_data(self):
        g1 = nx.karate_club_graph()
        g1.label = 0
        g2 = nx.complete_graph(13)
        g2.label = 1
        g3 = nx.dense_gnm_random_graph(10, 25, seed=42)
        g3.label = 0
        
        ret_val = [g1, g2, g3], 2, None
        
        with patch("backdoorpony.datasets.utils.graphbackdoor.load_data", return_value=ret_val):
            ret_val2 = [g1, g2], [g3], [2]
            with patch("backdoorpony.datasets.utils.graphbackdoor.separate_data", return_value=ret_val2):
                mutag = MUTAG()
                x1, y1 = mutag.get_data()
                self.assertTrue(x1[0] == [g1, g2])
                self.assertTrue(x1[1] == 2)
                self.assertTrue(y1[0] == [g3])
                self.assertTrue(y1[1] == [2])
                
                x2, y2 = mutag.get_datasets()
                self.assertTrue(x1[0] == [g1, g2])
                self.assertTrue(x2[1] == 2)
                self.assertTrue(y2[0] == [g3])
                self.assertTrue(y2[1] == [2])



if __name__ == '__main__':
    unittest.main()
