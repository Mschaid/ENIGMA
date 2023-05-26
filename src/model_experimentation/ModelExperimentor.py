from src.processors.Preprocessor import Preprocessor
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class ModelExperimentor(ABC):
    """Abstract class for model experimentation. The purpose of this class is to outline and constrain model experimentation that is implemented
    in the child classes that will be developed dependent on the framework being used (tensorflow, xgboost, pytorch). This class is not meant to be instantiated.

    Parameters
    ----------
    ABC : the abstract class from the abc module
    """
    def __init__(self, model, preprocessor: Preprocessor):
        """

        Parameters
        ----------
        model : The model to be experimented with, this is written outside of instantiation of the class
        preprocessor : a stored preprocessor object that stores preprocessed data for model experimentation
            
        """
        self.model = model
        self.preprocessor = preprocessor
        
    @abstractmethod
    def compile_model(self):
        """ calls the build method of the model object that is passed in to the class
        """
        pass
    
    @abstractmethod
    def train_model(self):
        """calls the train method of the model object that is passed in to the class
        """
        pass
    
    @abstractmethod
    def test_model(self):
        """calls the test method of the model object that is passed in to the class"""
        pass
    
    @abstractmethod
    def evaluate_model(self):
        """calls the evalutation methods of the model object that is passed in to the class
        """
        pass
    
    @abstractmethod
    def save_model(self):
        """ saves the model
        """
        pass
    

    
class TFModelExperimentor(ModelExperimentor):
    def __init__(self, model, preprocessor: Preprocessor):
        super().__init__(model, preprocessor)
        
    def compile_model(self, **kwargs):

        self.model.build()
        self.model.compile(**kwargs)
        return self

        
    def train_model(self):
        
        self.model.fit(
            self.preprocessor.X_train, 
            self.preprocessor.y_train,
            batch_size=10,
            epochs=10
        )
        return self
    def test_model(self):
        pass
    
    def evaluate_model(self):
        pass
    
    def save_model(self):
        pass