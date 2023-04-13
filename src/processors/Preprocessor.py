
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# todo logger
# todo unit tests


class Preprocessor:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def load_data(self):
        self.data = pd.read_parquet(self.path_to_data)
        return self

    def one_hot_encode(self, *labels, df=None):
        if df is None:
            df = self.data

        dataframes_w_dummies = [pd.get_dummies(
            df[label], prefix=label) for label in labels]
        concat_df = pd.concat([df, *dataframes_w_dummies], axis=1)
        packed_labels = list(labels)
        self.dummy_data = concat_df.drop(columns=packed_labels)

        return self

 
    def save_features(self, path):
        self.dummy_data.to_parquet(path)
