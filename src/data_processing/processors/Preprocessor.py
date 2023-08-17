
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
                 path_to_data: str = None,
                 path_to_processed_data: str = None,
                 path_to_save: str = None,
                 features: List[str] = None,
                 target: str = None,
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
        self.path_to_processed_data_dir = create_new_directory(
            'processed_data', root_dir=self.path_to_save)
        self.path_to_save_processor = create_new_directory(
            'processors', root_dir=self.path_to_save)
        self.path_to_save_datasets = create_new_directory(
            'datasets', root_dir=self.path_to_save)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.target = target
        self._all_data = self.features + [self.target]
        self._is_downsampled = False

    def load_data(self, load_processed_data: str = False) -> Type['Preprocessor']:
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
                raise FileTypeError(
                    'Filetype must be a parquet file and gzipped. Please provide a valid path to a parquet gzi')
            else:
                pass

        check_parquet_file(self.path_to_data)

        if self.path_to_data is not None:
            self._data = pd.read_parquet(self.path_to_data)
            self.data = self._data

        if load_processed_data == True and self.path_to_processed_data is not None:
            self._processed_data = pd.read_parquet(self.path_to_processed_data)
            self.processed_data = self._processed_data.dropna()

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
            data = self.processed_data[self._all_data]

        under_cut_off_data = data.query(
            f"{column_name} <= {cutoff}")

        over_cut_off_data = data.query(
            f"{column_name} > {cutoff}")

        self.X_train = under_cut_off_data[self.features]
        self.y_train = under_cut_off_data[self.target]

        self.X_test = over_cut_off_data[self.features]
        self.y_test = over_cut_off_data[self.target]

    # DATA PREPERATION METHODS

    def save_datasets_to_parquet(self, path=None, save_downsampled=False):
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
            (f'raw_data_{self.processor_name}',
             self.data), path_to_save=path)

        if self.X_train is not None:
            save_dataframes_to_parquet(
                (f'{self.processor_name}_X_train', self.X_train),
                (f'{self.processor_name}_X_test', self.X_test),
                (f'{self.processor_name}_y_train', self.y_train),
                (f'{self.processor_name}_y_test', self.y_test),
                path_to_save=path)

        if save_downsampled == True:
            save_dataframes_to_parquet(
                (f'{self.processor_name}_X_train_downsampled',
                 self.X_train_downsampled),
                (f'{self.processor_name}_X_test_downsampled',
                 self.X_test_downsampled),
                (f'{self.processor_name}_y_train_downsampled',
                 self.y_train_downsampled),
                (f'{self.processor_name}_y_test_downsampled',
                 self.y_test_downsampled),
                path_to_save=path)
        return self

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
