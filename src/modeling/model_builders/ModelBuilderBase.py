from src.processors.Preprocessor import Preprocessor
from abc import ABC, abstractmethod


class ModelBuilderBase(ABC):
    """Abstract class for model experimentation. The purpose of this class is to outline and constrain model experimentation that is implemented
    in the child classes that will be developed dependent on the framework being used (tensorflow, xgboost, pytorch). 
    This class is not meant to be instantiated. 
    Use assumes data was preprocessed with the PreProcessor class

    Parameters
    ----------
    ABC : the abstract class from the abc module
    """

    def __init__(self, 
                 model,
                 path_to_processed_data,
                 ):
        """

        Parameters
        ----------
        model : The model to be experimented with, this is written outside of instantiation of the class
        preprocessor : a stored preprocessor object that stores preprocessed data for model experimentation

        """
        self.model = None
        self.path_to_processed_data = path_to_processed_data
        

    def load_preprocessed_data(self, downsampled = False):
        """
        loads the preprocessed data from the path that is passed in to the class
        
        Parameters
        ----------
        downsampled : bool, optional
            if True, loads any datasets containing 'downsampled' in file name,  defaults to False
        
        """
        
        # search for file_paths containing X_train, y_train, X_test, y_test
        
        
        #filter for parquet files
        
        # read in parquet files to attributes
      
            
        
    # @abstractmethod
    # def compile_model(self):
    #     """ calls the build method of the model object that is passed in to the class
    #     """
    #     pass

    # @abstractmethod
    # def train_model(self):
    #     """calls the train method of the model object that is passed in to the class
    #     """
    #     pass

    # @abstractmethod
    # def test_model(self):
    #     """calls the test method of the model object that is passed in to the class"""
    #     pass

    # @abstractmethod
    # def evaluate_model(self):
    #     """calls the evalutation methods of the model object that is passed in to the class
    #     """
    #     pass

    # @abstractmethod
    # def save_model(self):
    #     """ saves the model
    #     """
    #     pass



