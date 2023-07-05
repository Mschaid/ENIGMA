"""
experiment for testing if binned trials is an import feature. 
This is the first model used with the sex and latency data included. 
We will use the same data with and without the binned trials

"""

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from src.utilities.os_helpers import create_new_directoy

from src.models.StackedLSTM import StackedLSTM
from src.utilities.tensorflow_helpers import set_tensorboard


MAIN_DIR = r'/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset'
PROCESSOR_NAME = '5_day_training_full_dataset_'
TENSORBOARD_DIR = "/projects/p31961/ENIGMA/results/tesnorboard_logs/bind_trials_experiment_01"
MODEL_SAVE_DIR = "/projects/p31961/ENIGMA/results/models/experiments/bind_trails_experiment_01"


X_train_path = os.path.join(
    MAIN_DIR, f'{PROCESSOR_NAME}X_train.parquet.gzip')
X_test_path = os.path.join(
    MAIN_DIR, f'{PROCESSOR_NAME}X_test.parquet.gzip')
y_train_path = os.path.join(
    MAIN_DIR, f'{PROCESSOR_NAME}y_train.parquet.gzip')
y_test_path = os.path.join(
    MAIN_DIR, f'{PROCESSOR_NAME}y_test.parquet.gzip')


def read_data():
    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)
    y_train = pd.read_parquet(y_train_path)
    y_test = pd.read_parquet(y_test_path)
    return X_train, X_test, y_train, y_test


def train_LSTM(model_ID):
    create_new_directoy(model_ID, root_dir=MODEL_SAVE_DIR)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = read_data()
