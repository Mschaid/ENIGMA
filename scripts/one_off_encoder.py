import pandas as pd
import numpy as np

DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'


def encoder_and_overwright(path, column):
    df = pd.read_parquet(path)
    df = pd.get_dummies(column)
    df.to_parquet(path)
    print(df).head()


if __name__ == "__main__":
    encoder_and_overwright(DATA_PATH, 'mouse_id')
