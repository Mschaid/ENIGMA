from src.processors.Preprocessor import Preprocessor
from abc import ABC, abstractmethod


class ModelBuilderBase(ABC):
    """Abstract class for model experimentation. The purpose of this class is to outline and constrain model experimentation that is implemented
    in the child classes that will be developed dependent on the framework being used (tensorflow, xgboost, pytorch). This class is not meant to be instantiated.

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
        self.model = model
        self.path_to_processed_data = path_to_processed_data
        

    def load_preprocessed_data(self):
        """loads the preprocessed data from the path that is passed in to the class
        """
        with h5py.File(self.path_to_processed_data, 'r') as file:
            dataframe_group = file['dataframes']
            for key in dataframe_group.keys():
                dataset = dataframe_group[key]
                setattr(self, key, pd.DataFrame(dataset[:]))
            
            attrs_group = file['attrs']
            for key in attrs_group.keys():
                setattr(self, key, attrs_group[key].value)
        
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



