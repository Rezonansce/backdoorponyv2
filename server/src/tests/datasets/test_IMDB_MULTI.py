import unittest
import unittest
from unittest import TestCase
from unittest.mock import PropertyMock
from backdoorpony.datasets.utils.gta.datareader import DataReader, GraphData
from torch.utils.data import DataLoader
from unittest import mock

from backdoorpony.datasets.IMDB_MULTI import IMDB_MULTI


class TestDataLoader(TestCase):
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.DataReader")
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.GraphData")
    @mock.patch("torch.utils.data.DataLoader")
    def test_get_data(self, dataReader, GraphData, DataLoader):
        
        data = {"splits" : {"train" : [0, 1 , 2, 3, 4, 5, 6, 7], "test" : [8, 9]}}
        type(dataReader.return_value).data = PropertyMock(return_value=data)
        GraphData.return_value = "gdata"
        DataLoader.return_value = 42
        
        imdb = IMDB_MULTI()
        x, y = imdb.get_datasets(0.01)
                    
        self.assertEqual(len(x[0]), 1)
        self.assertTrue(isinstance(x[1],  DataReader))
        
        self.assertEqual(len(y[0]), 1)
        self.assertEqual(len(y[1]), 3)



if __name__ == '__main__':
    unittest.main()
