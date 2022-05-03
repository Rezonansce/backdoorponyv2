from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def __init__(self):
        '''Should initiate the model
        '''
        pass