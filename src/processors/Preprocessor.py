
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def load_data(self):
        self.data = pd.read_parquet(self.path_to_data)
        return self

    def one_hot_encode(self, *labels, df=None):
        if df is None:
            df = self.data
            print(df)

        dataframes_w_dummies = [pd.get_dummies(
            df[label], prefix=label) for label in labels]
        concat_df = pd.concat([df, *dataframes_w_dummies], axis=1)
        packed_labels = list(labels)
        self.dummy_data = concat_df.drop(columns=packed_labels)

        return self


if __name__ == '__main__':
    print('preprocessor file')
