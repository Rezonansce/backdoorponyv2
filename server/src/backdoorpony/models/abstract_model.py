from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def __init__(self):
        '''Should initiate the model
        '''

        pass

    @abstractmethod
    def get_opti(self):
        '''
        Sould return the optimizer of the model
        :return:
        '''
        pass

    @abstractmethod
    def get_criterion(self):
        '''
        Should return the loss criterion of the model
        :return:
        '''
        pass

    @abstractmethod
    def get_nb_classes(self):
        '''
        Should return the number of classes of the model
        :return:
        '''
        pass

    @abstractmethod
    def get_input_shape(self):
        '''
        Should return the input shape of the model
        :return:
        '''
        pass

    @abstractmethod
    def get_path(self):
        '''
        Should return the name of the pre-load state_dict of the model
        :return:
        '''
        pass