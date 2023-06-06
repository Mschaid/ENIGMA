
from src.processors.Preprocessor import Preprocessor
from src.utilities.os_helpers import save_dataframes_to_parquet


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from time import sleep
import datetime

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Lambda


def read_data(path):
    """
    Read data from a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The path to the Parquet file.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    data = pd.read_parquet(path)
    return data


def split_data_by_trial(df, trial_threshold):
    """
    Split data into training and testing sets based on a trial threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    trial_threshold : int
        The threshold for splitting the data into trials.

    Returns
    -------
    tuple
        A tuple containing X_train, y_train, X_test, and y_test.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing labels.
    """
    trial_threshold = trial_threshold
    trials_under_5 = df.query('trial<=@trial_threshold')
    trials_over_5 = df.query('trial>@trial_threshold')

    X_train, y_train = trials_under_5.drop(
        columns=['signal']), trials_under_5.signal
    X_test, y_test = trials_over_5.drop(
        columns=['signal']), trials_over_5.signal
    return X_train, y_train, X_test, y_test


def build_lstm(sequence_length, input_dimentions):
    # TODO need to redo model architecture
    """
    Build a sequential model with a specific architecture.

    Returns
    -------
    tensorflow.keras.models.Sequential
        The built sequential model.
    """

    input_shape = (sequence_length, input_dimentions)
    ltsm_model = Sequential([
        Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    return ltsm_model


def set_tensorboard(model_id):
    """
    Set up TensorBoard for model training. Returns the TensorBoard callback.

    Parameters
    ----------
    model_id : str
        The ID of the model.

    Returns
    -------
    tensorflow.keras.callbacks.TensorBoard
        The TensorBoard callback.
    """
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_log_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/"
    logs_dir = f"{training_log_dir}/{model_id}/{date_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir, histogram_freq=1)
    return tensorboard_callback


def train_model(model, X_train, y_train, tensorboard_callback):
    # TODO save checkpoints
    """
    Train a model on the given training data. The optimizer and loss function are currently hard-coded.

    Parameters
    ----------
    model : tensorflow.keras.models.Sequential
        The model to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    tensorboard_callback : tensorflow.keras.callbacks.TensorBoard
        TensorBoard callback for logging.

    Returns
    -------
    None
    """
    model.compile(
        optimizer="adam", loss='mean_squared_error')
    model.fit(X_train, y_train, epochs= 50,
              callbacks=[tensorboard_callback])


def evaluate_model(model, X_test, y_test, callbacks=None):
    """
    Evaluate the trained model on the given testing data.

    Parameters
    ----------
    model : tensorflow.keras.models.Sequential
        The trained model.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing labels.

    Returns
    -------
    None
    """
    model.evaluate(X_test, y_test, callbacks = callbacks)


def inference(model, X_test):
    """
    Perform inference using the trained model on the given test data.

    Parameters
    ----------
    model : tensorflow.keras.models.Sequential
        The trained model.
    X_test : pd.DataFrame
        Testing features.

    Returns
    -------
    None
    """
    model.predict(X_test)

# def save_model(model, path_to_save, model_id):
#     """
#     Save the trained model to a specific path.

#     Parameters
#     ----------
#     model : tensorflow.keras.models.Sequential
#         The trained model.
#     path_to_save : str
#         The path to save the model.
#     model_id : str
#         The ID of the model.

#     Returns
#     -------
#     None
#     """
#     #TODO need to implement saving with keras
#     path = os.path.join(path_to_save, model_id)
#     save(os.path.join(path_to_save, model_id))


def validated_tf():
    """
    Validate TensorFlow installation and GPU availability. Prints the TensorFlow version and the GPUs available.

    Returns
    -------
    None
    """
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        print("GPU not available")
    else:
        for gpu in gpus:
            print(gpu)


def main():
    DATA_PATH = '/projects/p31961/dopamine_modeling/data/prototype_data/mouse_909_DA_avoid.parquet.gzip'
    MODEL_PATH_SAVE = '/projects/p31961/dopamine_modeling/results/models/'
    validated_tf()
    data = read_data(DATA_PATH)
    model_id = 'ltsm_prototype_tensor_datasets'
    X_train, y_train, X_test, y_test = split_data_by_trial(data, 5)
    X_train_ds = X_train[::100]
    y_train_ds = y_train[::100]
    X_test_ds = X_test[::100]
    y_test_ds = y_test[::100]


    save_dataframes_to_parquet(
        ('X_train', X_train),
        ('X_test', X_test),
        ('y_train', y_train),
        ('y_test', y_test), path_to_save='/projects/p31961/dopamine_modeling/data/prototype_data')

    
    time window is 45 seconds, we will use 90 sequence length for 1/2 second per sequence
    model = build_lstm(sequence_length=90, input_dimentions=X_train_ds.shape[1])
    tensorboard_callback = set_tensorboard(model_id)
    train_model(model, X_train_ds, y_train_ds, tensorboard_callback)
    evaluate_model(model, X_test_ds, y_test_ds, tensorboard_callback)
    inference(model, X_test_ds)
    tf.keras.models.save_model(model, os.path.join(MODEL_PATH_SAVE, model_id))
    print(X_train_ds)


if __name__ == "__main__":
    main()
