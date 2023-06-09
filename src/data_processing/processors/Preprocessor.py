
import h5py
import os
import pandas as pd
from typing import List, Type
import _pickle as cPickle
import numpy as np
from src.utilities.os_helpers import *
from src.utilities.exceptions import *


from sklearn.model_selection import train_test_split


class Preprocessor:

    """
    Summary
    ----------
    Preprocessor is responsible for loading,  preprocessing, and saving processed the data. for model building and experimentation


    Attributes
    ----------
     processor_name : str or None
                The name of the processor.
    path_to_data : str or None
        The path to the raw data file.
    path_to_processed_data : str or None
        The path to the processed data file.
    features : List[str] or None
        A list of column names that correspond to the features in the dataset.
    target : str or None
        The name of the target variable in the dataset.
    X_train : pd.DataFrame or None
        The training features data.
    X_test : pd.DataFrame or None
        The test features data.
    y_train : pd.Series or None
        The training target data.
    y_test : pd.Series or None
        The test target data.


    Methods
    ----------
    load_data
        loads data from path_to_data or path_to_processed_data into data attribute. If both attributes are not None, both are loaded.
    one_hot_encode
        encodes the data using one hot encoding. 
    split_train_test
        splits data into train and test sets using train_test_split from sklearn.model_selection

    split_train_by_query
        spilts data into train and test sets based on a specified column's cutoff value.
    save_processed_data
        saves processed data attribute to path_to_processed_data
    load_processed_data
        class method that loads an instance of Preprocessor from a pickle file.    
    """

    def __init__(self,
                 processor_name: str = None,
                 path_to_data: str =None,
                 path_to_processed_data: str =None,
                 path_to_save:str = None,
                 features: List[str] =None,
                 target:str =None,
                 X_train: pd.DataFrame = None,
                 X_test: pd.DataFrame = None,
                 y_train: pd.Series = None,
                 y_test: pd.Series = None):
        """# Summary
        Instantiates a Preprocessor object.

        ### Args:
            path_to_data (_type_): _description_
        """
        self.processor_name = processor_name
        self.path_to_data = path_to_data
        self.path_to_save = create_dir(path_to_save)
        self.path_to_processed_data = path_to_processed_data
        self.path_to_processed_data_dir = create_new_directoy(
            path_to_save, 'processed_data')
        self.path_to_save_processor = create_new_directoy(
            path_to_save, 'processors')
        self.path_to_save_datasets = create_new_directoy(
            path_to_save, 'datasets')
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.target = target
        self._is_downsampled = False

    def load_data(self, load_processed_data: str =False) -> Type['Preprocessor']:
        """
        # Summary
        loads data from path_to_data into data attribute.
        first checks that filetype is parquet (currently only supported filetype)


        # attributes
        stores data from path_to_data into data attribute as a Pd.DataFrame object.
        
        Returns:
            self
        """
        #! needs tesing
        def check_parquet_file(path):
            if path is not None and not path.endswith('.gzp'):
                raise FileTypeError('Filetype must be a parquet file and gzipped. Please provide a valid path to a parquet gzi')
            else: 
                pass
            
        check_parquet_file(self.path_to_data)
        
        if self.path_to_data is not None:
            self._data = pd.read_parquet(self.path_to_data)
            self.data = self._data.dropna()

        if load_processed_data == True and self.path_to_processed_data is not None:
            self._processed_data = pd.read_parquet(self.path_to_processed_data)
            self.processed_data = self._processed_data.dropna()

        return self
    # FEATURE ENGINEERING METHODS
    def down_sample_dataframe(self, group_by_columns:str, sort_by_cols:str, agg_dict:dict, n:int) -> Type['Preprocessor']:
        pass

                    
        
    def one_hot_encode(self, labels: List[str], data: pd.DataFrame = None):
        """# Summary

        ## Args:
            - df (pd.DataFrame, optional): dataframe . Defaults to self.data.
            - labels (list, optional): list of column names to one hot encode, these columns are prefixed to the name, and original columns are dropped.
        ## Attributes:
        - processed_data: pd.DataFrame this is the processed encoded data that joined with the original data- ready for model input
        - labels: list of column names to one hot encode, these columns are prefixed to the name, and original columns are dropped.
        Returns:
            self
        """
        if data is None:
            data = self.data

        # dataframes_w_dummies = [pd.get_dummies(
        #     data[label], prefix=label) for label in labels]
        # concat_df = pd.concat([data, *dataframes_w_dummies], axis=1)
        # packed_labels = list(labels)
        # self.dummy_data = concat_df.drop(columns=packed_labels)

        self.processed_data = pd.get_dummies(data=data,
                                             prefix=labels,
                                             columns=labels)
        return self

    def split_train_test(self, test_size=0.2, random_state=42, processed_data=False, data=None):
        """

    Summary
    -------
    Split the processed data into training and testing sets using the train_test_split method from scikit-learn.

    Parameters
    ----------
    - test_size (float, optional): the proportion of the data to use for testing, should be between 0 and 1.
      Defaults to 0.2 (i.e., 20% of the data is used for testing).
    - random_state (int, optional): controls the randomness of the data splitting process.
      Defaults to 42, which means that the same random seed will be used every time the method is called,
      ensuring reproducibility of the results.

    Returns:
    -------
    - self: the class instance with attributes X_train, X_test, y_train, and y_test.
      X_train and y_train are the training features and target data, while X_test and y_test are the testing
      features and target data.
        """
        if processed_data is True:
            data = self.processed_data

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data[self.features],
            data[self.target],
            test_size=test_size,
            random_state=random_state)

        return self

    def split_train_by_query(self, column_name, cutoff, processed_data=False, data=None):
        """
        Summary
        -------
        Split the processed data into training and testing sets based on a specified column's cutoff value.

        Parameters
        ----------
        column_name : str
            Name of the column to split the data on.
        cutoff : float
            The cutoff value to use to split the data. Any value less than or equal to the cutoff
            will be in the training set, while any value greater than the cutoff will be in the testing set.

        Returns
        -------
        None

        Notes:
        ------
            Updates the class instance with four attributes - X_train, X_test, y_train, and y_test.
            X_train and y_train are the training features and target data, respectively, which consist of rows from
            the processed data where the value of the specified column is less than or equal to the cutoff.
            X_test and y_test are the testing features and target data, respectively, which consist of rows from the
            processed data where the value of the specified column is greater than the cutoff.
        """

        if processed_data is True:
            data = self.processed_data

        under_cut_off_data = data.query(
            f"{column_name} <= {cutoff}")

        over_cut_off_data = data.query(
            f"{column_name} > {cutoff}")

        self.X_train = under_cut_off_data[self.features]
        self.y_train = under_cut_off_data[self.target]

        self.X_test = over_cut_off_data[self.features]
        self.y_test = over_cut_off_data[self.target]
    
    #DATA PREPERATION METHODS

    def save_datasets_to_parquet(self, path=None, save_downsampled = False):
        """
        saves the processed datasets (X_train, X_test, y_train, y_test, to parquet files
        Parameters
        ----------
        path : str, optional 

        Attributes
        ----------
        path_to_save_datasets : str is path is provided, path_to_save_datasets is updated, 
                                if not defaults to initialized self.path_to_save_datasets

        """

        if path is not None:
            self.path_to_save_datasets = path
        else:
            path = self.path_to_save_datasets
            
          
        # create_dir(self.path_to_save_datasets)
        save_dataframes_to_parquet(
            (f'{self.processor_name}_X_train', self.X_train),
            (f'{self.processor_name}_X_test', self.X_test),
            (f'{self.processor_name}_y_train', self.y_train),
            (f'{self.processor_name}_y_test', self.y_test),
            path_to_save=path)
        
        if save_downsampled == True:
            save_dataframes_to_parquet(
                (f'{self.processor_name}_X_train_downsampled', self.X_train_downsampled),
                (f'{self.processor_name}_X_test_downsampled', self.X_test_downsampled),
                (f'{self.processor_name}_y_train_downsampled', self.y_train_downsampled),
                (f'{self.processor_name}_y_test_downsampled', self.y_test_downsampled),
                path_to_save=path)
        return self
    #! currently not implemented - should likely be removed
    def save_data_to_h5(self, path=None, file_name=None):
        """
        method to save specific attributes of the processor object to an hdf5 file

        Parameters
        ----------
        path : str, optional - path to save the hdf5 file to. Defaults to path_to_save_datasets
        file_name: str, optional - name of the hdf5 file to save the data to. Defaults to <processor_name>_dataset.h5

        Attributes saved
        ----------------
        * processor_name
        * path_to_data
        * path_to_save
        * path_to_processed_data
        * path_to_processed_data_dir
        * path_to_save_processor
        * path_to_save_datasets
        * features list
        * target
        * X_train
        * X_test
        * y_train
        * y_test

        Returns
        -------
        None
        """

        if path is not None:
            self.path_to_save_datasets = path
        else:
            path = self.path_to_save_datasets

        if file_name is None:
            file_name = f'{self.processor_name}_dataset.h5'
            
        file_path = os.path.join(path, file_name)

        dataframe_dict = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
            
        }

        attrs_dict = {
            'processor_name': str(self.processor_name),
            'path_to_data': str(self.path_to_data),
            'path_to_save': str(self.path_to_save),
            'path_to_processed_data': str(self.path_to_processed_data),
            'path_to_processed_data_dir': str(self.path_to_processed_data_dir),
            'path_to_save_processor': str(self.path_to_save_processor),
            'path_to_save_datasets': str(self.path_to_save_datasets),
            'features': self.features,
            'target': self.target
        }
        
        def save_to_h5(file_path, dataframe_dict, attrs_dict):
                with h5py.File(file_path, 'w') as file:
                    # save dataframes 
                    dataframe_group = file.create_group('dataframes')
                    for key, value in dataframe_dict.items():
                        dataframe_group[key] = value
                    # save attributes
                    attrs_group = file.create_group('attributes')
                    for key, value in attrs_dict.items():
                        attrs_group[key] = value    
                        
        save_to_h5(file_path, dataframe_dict, attrs_dict)

            # ? possible implementiation but need to resolve speed issues
    
     #! currently not implemented - should likely be removed
   
    def save_processor(self, path_to_save_processor=None):
        """
        Summary
        -------
        Saves the processed data to a file.


        Parameters
        ---
        path: str path to save the processed data to.

        Notes:
        ------
        The method creates a new directory called `processors` in the same directory as
        the processed data, if it doesn't exist already. It then saves the processor object
        to a pickle file with the same name as the processor, using the following format:

            <processed_data_directory>/processors/<processor_name>.pkl

        Returns:
            None

        """
        if path_to_save_processor is not None:
            self.path_to_save_processor = path_to_save_processor
        else:
            # gets the directory of the processed data path
            current_directory = os.path.dirname(self.path_to_data)
            # creates a new directory called processors in the same directory as 1the processed data
            processor_directory = os.path.join(current_directory, 'processors')

            self.path_to_save_processor = os.path.join(
                processor_directory, f'{self.processor_name}.pkl')

        # check if the processor directory exists, if not, create it
        create_dir(processor_directory)

        # save the processor
        with open(self.path_to_save_processor, 'wb') as f:
            cPickle.dump(self, f)

    #! currently not implemented - should likely be removed
    def save_processor_to_h5(self):
        
        '''self.filename = os.path.join(self.path_to_save_processor, f'{self.processor_name}.h5')
        with h5py.File(self.filename, 'w') as file:
            instance_group = file.create_group('instance')

            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, pd.DataFrame):
                    # Save DataFrame as a separate group
                    df_group = instance_group.create_group(attr_name)
                    for col_name, col_data in attr_value.items():
                        df_group.create_dataset(col_name, data=col_data)
                elif isinstance(attr_value, (int, float, str)):
                    instance_group.create_dataset(attr_name, data=attr_value)
                else:
                    # Convert non-compatible types to string representation
                    instance_group.create_dataset(attr_name, data=str(attr_value))'''
        pass

    #! currently not implemented
    @classmethod
    def load_processor(cls, file_path):
        """ load a saved processor object from a pickle file

        Parameters
        ----------
        file_path : str
            the path to the pickle file containing the saved processor object
        """

        def generator():
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        yield cPickle.load(f)
                    except EOFError:
                        break

        processor = cls()
        for obj in generator():
            processor = obj
        return processor
