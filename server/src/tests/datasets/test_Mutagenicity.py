import unittest
from unittest import TestCase
from unittest.mock import PropertyMock
from backdoorpony.datasets.utils.gta.datareader import DataReader, GraphData
from torch.utils.data import DataLoader
from unittest import mock

from backdoorpony.datasets.Mutagenicity import Mutagenicity


class TestDataLoader(TestCase):
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.DataReader")
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.GraphData")
    @mock.patch("torch.utils.data.DataLoader")
    def test_get_data(self, dataReader, GraphData, DataLoader):
        
        data = {"splits" : {"train" : [0, 1 , 2, 3, 4, 5, 6, 7], "test" : [8, 9]}}
        type(dataReader.return_value).data = PropertyMock(return_value=data)
        GraphData.return_value = "gdata"
        DataLoader.return_value = 42
        
        mutagenicity = Mutagenicity(0.01)
        x1, y1 = mutagenicity.get_data()
        
        self.assertEqual(len(x1[0]), 2)
        self.assertTrue(isinstance(x1[1],  DataReader))
                    
        x2, y2 = mutagenicity.get_datasets()
        self.assertEqual(len(x2[0]), 2)
        self.assertTrue(isinstance(x2[1],  DataReader))
        
        self.assertEqual(len(y1[0]), 14)
        self.assertEqual(len(y1[1]), 434)
        self.assertEqual(len(y2[0]), 14)
        self.assertEqual(len(y2[1]), 434)



if __name__ == '__main__':
    unittest.main()

