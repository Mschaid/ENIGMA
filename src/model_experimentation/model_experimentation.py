
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
    data = pd.read_parquet(path)
    return data

def split_data_by_trial(df, trial_threshold):
    trial_threshold = trial_threshold
    trials_under_5 = df.query('trial<=@trial_threshold')
    trials_over_5 = df.query('trial>@trial_threshold')

    X_train, y_train = trials_under_5.drop(columns = ['signal']), trials_under_5.signal
    X_test, y_test = trials_over_5.drop(columns = ['signal']), trials_over_5.signal
    return X_train, y_train, X_test, y_test
        

def build_model():
    #TODO need to redo model architecture
    sequential_model = Sequential(
    [
        Dense(128, activation='relu', name="dense_1"),
        Dense(1, activation='relu', name="dense_2")
    ]
)
    return sequential_model

def set_tensorboard(model_id):
    model_id = 'sequential_prototype'
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_log_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/" 
    logs_dir = f"{training_log_dir}/{model_id}/{date_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
    return tensorboard_callback
        
def train_model(model, X_train, y_train, tensorboard_callback):
    model.compile(
    optimizer="adam", loss=Huber())
    model.fit(X_train, y_train, batch_size=30, epochs= 100, callbacks=[tensorboard_callback])

def evaluate_model(model, X_test, y_test):
    model.evaluate(X_test, y_test)
    
def inference(model, X_test):
    model.predict(X_test)
    
def save_model(model, path_to_save, model_id):
    #TODO need to implement saving with keras
    model.save(os.path.join(path_to_save, model_id))
    

    
def validated_tf():
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
    
