from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def __init__(self):
        '''Should initiate the model
        '''

        pass

    @abstractmethod
    def get_opti(self):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    @abstractmethod
    def get_nb_classes(self):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass
