import unittest
from unittest import TestCase
from unittest.mock import PropertyMock
from backdoorpony.datasets.utils.gta.datareader import DataReader, GraphData
from torch.utils.data import DataLoader
from unittest import mock

from backdoorpony.datasets.AIDS import AIDS


class TestDataLoader(TestCase):
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.DataReader")
    @mock.patch("backdoorpony.datasets.utils.gta.datareader.GraphData")
    @mock.patch("torch.utils.data.DataLoader")
    def test_get_data(self, dataReader, GraphData, DataLoader):
        
        data = {"splits" : {"train" : [0, 1 , 2, 3, 4, 5, 6, 7], "test" : [8, 9]}}
        type(dataReader.return_value).data = PropertyMock(return_value=data)
        GraphData.return_value = "gdata"
        DataLoader.return_value = 42
        
        aids = AIDS()
        x1, y1 = aids.get_data()
        self.assertTrue(len(x1[0]) == 50)
        self.assertTrue(len(x1[1]) == 13)
                    
        x2, y2 = aids.get_datasets()
        self.assertTrue(len(x2[0]) == 50)
        self.assertTrue(len(x2[1]) == 13)
        
        self.assertTrue(isinstance(y1,  DataReader))
        self.assertTrue(isinstance(y2,  DataReader))



if __name__ == '__main__':
    unittest.main()

