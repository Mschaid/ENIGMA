
import pandas as pd
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

    def __init__(self, path_to_data):
        """# Summary
        Instantiates a Preprocessor object.

        ### Args:
            path_to_data (_type_): _description_
        """
        self.path_to_data = path_to_data

    def load_data(self):
        """
        # Summary
        loads data from path_to_data into data attribute.


        # attributes
        stores data from path_to_data into data attribute as a Pd.DataFrame object.

        Returns:
            self
        """
        self.data = pd.read_parquet(self.path_to_data)
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

    def save_features(self, path):
        """
        # Summary
        saves the processed data to path.

        ### Args:
            - path (str): saves data to this path.
        """
        self.dummy_data.to_parquet(path)
