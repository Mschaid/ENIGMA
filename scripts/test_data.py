import logging
import numpy as np
import os
import pandas as pd


from src.utilities.os_helpers import create_new_directory, create_directories
from src.utilities.pandas_helpers import get_features


DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'


def split_data(data, features, target, day_cut_off, feature_to_drop=None):

    # drops feature prior to splitting data
    if feature_to_drop is not None:
        data = data.drop(columns=feature_to_drop)

    day_cut_off = day_cut_off

    training_set = data.query('day < @day_cut_off')
    valdidation_set = data.query('day == @day_cut_off')
    testing_set = data.query('day > @day_cut_off')

    X_train, y_train = training_set[features], training_set[target]
    X_val, y_val = valdidation_set[features], valdidation_set[target]
    X_test, y_test = testing_set[features], testing_set[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


data = pd.read_parquet(DATA_PATH)
target = 'signal'
control_features = get_features(data, target)
no_bin_features = control_features.copy()
no_bin_features.remove('learning_phase')

# split data for control
logging.info('Splitting data into training, validation and testing sets')
X_train, y_train, X_val, y_val, X_test, y_test = split_data(
    data, control_features, target, day_cut_off=6, feature_to_drop=None)

# split data for dropped binned trials
X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped, X_test_dropped, y_test_dropped = split_data(
    data, no_bin_features, target, day_cut_off=6, feature_to_drop='learning_phase')

print('X data shapes')
print('control: ', X_train.shape, X_val.shape, X_test.shape)
print('dropped: ', X_train_dropped.shape,
      X_val_dropped.shape, X_test_dropped.shape)

print('y data shapes')
print('control: ',  y_train.shape, y_val.shape, y_test.shape)
print('dropped: ', y_train_dropped.shape,
      y_val_dropped.shape, y_test_dropped.shape)

print('all data shapes:', data.shape)
