
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

    def split_train_test(self, X_labels, y_label, test_size=0.2, random_state=42):
        self.features = self.dummy_data[X_labels]
        self.target = self.dummy_data[y_label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state)
        return self
