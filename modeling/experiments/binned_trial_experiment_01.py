"""
experiment for testing if binned trials is an import feature. 
This is the first model used with the sex and latency data included. 
We will use the same data with and without the binned trials

"""
import atexit
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from src.utilities.os_helpers import create_new_directory, create_directories
from src.utilities.pandas_helpers import get_features
from src.models.StackedLSTM import StackedLSTM
from src.utilities.tensorflow_helpers import set_tensorboard


MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
EXPERIMENT_NAME = 'binned_trial_experiment_01'

EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')
# global variables
DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'
TENSORBOARD_DIR = os.path.join(EXPERIMENT_DIR, 'tensorboard')
MODEL_SAVE_DIR = os.path.join(EXPERIMENT_DIR, 'models')

# clear log file
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s - %(message)s')


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


def train_LSTM(model_ID, X_train, y_train, X_val, y_val, X_test, y_test):
    tensoboard_callback = set_tensorboard(model_ID, TENSORBOARD_DIR)

    model = StackedLSTM(sequence_length=90,
                        num_features=X_train.shape[1],
                        lstm_units=128)
    model.compile(optimizer='adam', loss='mse', metrics=[
                  'mae', 'mse', 'mape', 'cosine_similarity'])
    model.fit(X_train, y_train,
              epochs=50,
              validation_data=(X_val, y_val),
              callbacks=[tensoboard_callback])
    model.evaluate(X_test, y_test)
    model.save(os.path.join(MODEL_SAVE_DIR, model_ID))

    create_new_directory(model_ID, root_dir=MODEL_SAVE_DIR)


def experiment():
    new_dirs = [EXPERIMENT_DIR, TENSORBOARD_DIR, MODEL_SAVE_DIR]

    for dirs in new_dirs:
        os.makedirs(dirs, exist_ok=True)

    logging.info('New directories created')

    logging.info('Loading data')
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

    logging.info('Training control model')
    # train model on control
    train_LSTM(model_ID='control',
               X_train=X_train, y_train=y_train,
               X_val=X_val, y_val=y_val,
               X_test=X_test, y_test=y_test)
    logging.info('Control model training complete')
    # train model on dropped binned trials

    logging.info('Training dropped binned trials model')
    train_LSTM(model_ID='dropped_binned_trials',
               X_train=X_train_dropped, y_train=y_train_dropped,
               X_val=X_val_dropped, y_val=y_val_dropped,
               X_test=X_test_dropped, y_test=y_test_dropped)
    logging.info('Dropped binned trials model training complete')


@atexit.register
def early_termination():
    logging.info('Experiment terminated early')


if __name__ == '__main__':

    logging.info('Starting experiment')
    experiment()
    logging.info('Experiment complete')
