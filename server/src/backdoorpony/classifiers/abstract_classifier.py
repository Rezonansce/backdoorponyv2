from abc import ABC, abstractmethod


class AbstractClassifier(ABC):

    @abstractmethod
    def __init__(self, model):
        '''Should initiate the classifier

        Parameters
        ----------
        model :
            Model that the classifier should be based on

        Returns
        ----------
        None
        '''
        pass

    @abstractmethod
    def fit(self, x, y, *args, **kwargs):
        '''Should fit the classifier to the training data
        
        Parameters
        ----------
        x :
            Data that the classifier will be trained on
        y :
            Labels that the classifier will be trained on

        Returns
        ----------
        None
        '''
        pass

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        '''Should return the predicted classification of the input

        Parameters
        ----------
        x :
            The dataset the classifier should classify

        Returns
        ----------
        prediction : 
            Return format can be anything, as long as it is consistent between
            classifiers of the same category
        '''
        pass

    @abstractmethod
    def class_gradient(self, x, *args, **kwargs):
        '''
        ...
        '''
        pass
