
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# todo logger
# todo unit tests


class Preprocessor:
    """
    # Summary
    Preprocessor is responsible for loading,  preprocessing, and saving processed the data. for model building and experimentation

    ## Attributes
    - path_to_data: str this is the path to the aggregated data. Must be a parquet file. This is stored when class is initialized.
    - data: pd.DataFrame this is the data that is loaded.
    - dummy_data: pd.DataFrame this is the processed encoded data that joined with the original data- ready for model input

    """

    def __init__(self,
                 processor_name=None,
                 path_to_data=None,
                 path_to_processed_data=None,
                 features=None,
                 target=None,
                 X_train=None,
                 X_test=None,
                 y_train=None,
                 y_test=None):
        """# Summary
        Instantiates a Preprocessor object.

        ### Args:
            path_to_data (_type_): _description_
        """
        self.processor_name = processor_name
        self.path_to_data = path_to_data
        self.path_to_processed_data = path_to_processed_data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.features = features
        self.target = target

    def load_data(self):
        """
        # Summary
        loads data from path_to_data into data attribute.


        # attributes
        stores data from path_to_data into data attribute as a Pd.DataFrame object.

        Returns:
            self
        """
        if self.path_to_data is not None:
            self.data = pd.read_parquet(self.path_to_data)

        if self.path_to_processed_data is not None:
            self.processed_data = pd.read_parquet(self.path_to_processed_data)

        return self

    def one_hot_encode(self, *labels, df=None):
        """# Summary

        ## Args:
            - df (pd.DataFrame, optional): dataframe . Defaults to self.data.
            - labels (list, optional): list of column names to one hot encode, these columns are prefixed to the name, and original columns are dropped.
        ## Attributes:
        - dummy_data: pd.DataFrame this is the processed encoded data that joined with the original data- ready for model input
        - labels: list of column names to one hot encode, these columns are prefixed to the name, and original columns are dropped.
        Returns:
            self
        """
        if df is None:
            df = self.data

        dataframes_w_dummies = [pd.get_dummies(
            df[label], prefix=label) for label in labels]
        concat_df = pd.concat([df, *dataframes_w_dummies], axis=1)
        packed_labels = list(labels)
        self.dummy_data = concat_df.drop(columns=packed_labels)

        return self

    def split_train_test(self, test_size=0.2, random_state=42):
        """

    # Summary
    Split the processed data into training and testing sets using the train_test_split method from scikit-learn.

    ## Args:
    - test_size (float, optional): the proportion of the data to use for testing, should be between 0 and 1.
      Defaults to 0.2 (i.e., 20% of the data is used for testing).
    - random_state (int, optional): controls the randomness of the data splitting process.
      Defaults to 42, which means that the same random seed will be used every time the method is called,
      ensuring reproducibility of the results.

    ## Returns:
    - self: the class instance with attributes X_train, X_test, y_train, and y_test.
      X_train and y_train are the training features and target data, while X_test and y_test are the testing
      features and target data.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.processed_data[self.features],
            self.processed_data[self.target],
            test_size=test_size,
            random_state=random_state)
        return self

    def split_train_by_query(self, column_name, cutoff):
        """
        # Summary
         Splits the processed data into training and testing sets based on a specified column's cutoff value using
         Pandas query method.

        ## Args:
        - column_name (str): name of the column to split the data on.
        - cutoff (float): the cutoff value to use to split the data, any value less than or equal to the cutoff
        will be in the training set, while any value greater than the cutoff will be in the testing set.

        ## Returns:
        - None: updates the class instance with four attributes - X_train, X_test, y_train, and y_test.
        X_train and y_train are the training features and target data, respectively, which consist of rows from
        the processed data where the value of the specified column is less than or equal to the cutoff.
        X_test and y_test are the testing features and target data, respectively, which consist of rows from the
        processed data where the value of the specified column is greater than the cutoff.
        """

        under_cut_off_data = self.processed_data.query(
            f"{column_name} <= {cutoff}")

        over_cut_off_data = self.processed_data.query(
            f"{column_name} > {cutoff}")

        self.X_train = under_cut_off_data[self.features]
        self.y_train = under_cut_off_data[self.target]

        self.X_test = over_cut_off_data[self.features]
        self.y_test = over_cut_off_data[self.target]

    def save_processed_data(self, path):
        """
        # Summary
        saves the processed data to path.

        ### Args:
            - path (str): saves data to this path.
        """
        self.dummy_data.to_parquet(path)

    def save_processor(self):
        """
        
        # Summary
        Saves the processor object to a file.

        The method creates a new directory called `processors` in the same directory as
        the processed data, if it doesn't exist already. It then saves the processor object
        to a pickle file with the same name as the processor, using the following format:

            <processed_data_directory>/processors/<processor_name>.pkl

        Returns:
            None
        
        """
     
        # gets the directory of the processed data path
        current_directory = os.path.dirname(self.path_to_processed_data)
        # creates a new directory called processors in the same directory as
        # the processed data
        processor_directory = os.path.join(current_directory, 'processors')
        
        path_to_save_processor = os.path.join(processor_directory, f'{self.processor_name}.pkl')

        # check if the processor directory exists, if not, create it
        if not os.path.exists(processor_directory):
            os.makedirs(processor_directory)
        
        # save the processor
        with open(path_to_save_processor, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_processor(cls, file_path):
        """
        # Summary
        loads the schema object from a pick file, can be loaded later. 

        Args:
            file_path (str):path for the file to be saved

        """
        with open(file_path, 'rb') as f:
            processor = pickle.load(f)
        return processor
if __name__ == "__main__":

    # FEATURES = ['day', 'time', 'trial',
    #             'signal', 'event_cue', 'event_shock',
    #             'sensor_D1', 'sensor_D2', 'sensor_DA']
    # TARGET = 'signal'
    # processor = Preprocessor(path_to_processed_data='/home/mds8301/gaby_test/processed_data.parquet.gzp',
    #                          features=FEATURES,
    #                          target=TARGET, 
    #                          processor_name='test_processor')

    # processor.load_data()
    # processor.split_train_test()
    # processor.save_processor()
    processor = Preprocessor.load_processor('/home/mds8301/gaby_test/processors/test_processor.pkl')
    print(processor.processor_name)
    print(processor.features)
    print(processor.X_train)