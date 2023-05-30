
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber

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

    X_train, y_train = trials_under_5.drop(columns = ['signal']), trials_under_5.signal
    X_test, y_test = trials_over_5.drop(columns = ['signal']), trials_over_5.signal
    return X_train, y_train, X_test, y_test
        

def build_model():
    #TODO need to redo model architecture
    """
    Build a sequential model with a specific architecture.

    Returns
    -------
    tensorflow.keras.models.Sequential
        The built sequential model.
    """
    sequential_model = Sequential(
    [
        Dense(128, activation='relu', name="dense_1"),
        Dense(1, activation='relu', name="dense_2")
    ]
)
    return sequential_model

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
    model_id = 'sequential_prototype'
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_log_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/" 
    logs_dir = f"{training_log_dir}/{model_id}/{date_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
    return tensorboard_callback
        
def train_model(model, X_train, y_train, tensorboard_callback):
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
    optimizer="adam", loss=Huber())
    model.fit(X_train, y_train, batch_size=30, epochs= 100, callbacks=[tensorboard_callback])

def evaluate_model(model, X_test, y_test):
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
    model.evaluate(X_test, y_test)
    
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
    
def save_model(model, path_to_save, model_id):
    """
    Save the trained model to a specific path.

    Parameters
    ----------
    model : tensorflow.keras.models.Sequential
        The trained model.
    path_to_save : str
        The path to save the model.
    model_id : str
        The ID of the model.

    Returns
    -------
    None
    """
    #TODO need to implement saving with keras
    model.save(os.path.join(path_to_save, model_id))
    

    
def validated_tf():
    """
    Validate TensorFlow installation and GPU availability. Prints the TensorFlow version and the GPUs available.

    Returns
    -------
    None
    """
    print("TensorFlow version:", tf.__version__)
    for gpu in tf.config.list_physical_devices('GPU'):
        print(gpu)


    
def main():
    DATA_PATH = '/projects/p31961/dopamine_modeling/data/prototype_data/mouse_909_DA_avoid.parquet.gzip'
    MODEL_PATH_SAVE = '/projects/p31961/dopamine_modeling/results/models/'
    validated_tf()
    data = read_data(DATA_PATH)
    model_id = 'sequential_prototype'
    X_train, y_train, X_test, y_test = split_data_by_trial(data, 5)
    
    save_dataframes_to_parquet(
        ('X_train', X_train),
        ('X_test', X_test),
        ('y_train', y_train),
        ('y_test', y_test)
        ,path_to_save='/projects/p31961/dopamine_modeling/data/prototype_data')
    
    
    model = build_model()
    tensorboard_callback = set_tensorboard(model_id)
    train_model(model, X_train, y_train, tensorboard_callback)
    evaluate_model(model, X_test, y_test)
    inference(model, X_test)
    save_model(model, path_to_save = MODEL_PATH_SAVE, model_id = model_id)
    
if __name__ == "__main__":
    main()
    
