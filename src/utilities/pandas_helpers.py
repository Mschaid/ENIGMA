import numpy as np
import pandas as pd
from typing import List


def flatten_dataframe(df):
    """Flatten a multi-indexed dataframe to a single index."""
    df.columns = ['_'.join(col) for col in df.columns.values]
    df = df.reset_index()
    return df


def strip_columns(df, strip_char: str = '_'):
    """
    Strip a character from the end of all column names.
    defaults to _ 
    """
    df.columns = [col.rstrip(strip_char) for col in df.columns]
    return df


def get_features(df, target):
    """ 
    Get a list of features from a dataframe, excluding the target column.
    """
    features = df.columns.tolist()
    features.remove(target)
    return features


def filter_columns_by_search(df: pd.DataFrame, search_term: str) -> List:
    filtered_cols = [col for col in df.columns if search_term in col]
    return filtered_cols
