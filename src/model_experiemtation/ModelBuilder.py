
from src.processors.Preprocessor import Preprocessor

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split
import logging

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy


class ModelBuilder:

    def __init__(self, path_to_processed_data, features, target, model_name, model):
        self.path_to_processed_data = path_to_processed_data
        self.features = features
        self.target = target
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_name = model_name
        self.model = model

    def load_data(self):
        self.processed_data = pd.read_parquet(self.path_to_processed_data)


    def save_model(self, model, model_name, path):
        model.save(os.path.join(path, model_name))

    def load_model(self, model_name, path):
        self.model_name = tf.keras.models.load_model(
            os.path.join(path, model_name))
        return self.model_name


def experiment():
    FEATURES = ['day', 'time', 'trial',
                'signal', 'event_cue', 'event_shock',
                'sensor_D1', 'sensor_D2', 'sensor_DA']
    TARGET = 'signal'

    PATH_TO_PROCESSED_DATA = input('Enter path to processed data: ')
    
    processor = Preprocessor(PATH_TO_PROCESSED_DATA,)
    
    

    fully_connected_model_1 = Sequential([
        Dense(units=128, activation='sigmoid'),
        Dense(units=25, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')])
    
    model_builder = ModelBuilder(path=PATH_TO_PROCESSED_DATA,
                                 features=FEATURES,
                                 target=TARGET,
                                 model_name='fully_connected_model_1',
                                 model=fully_connected_model_1)

    model_builder.model.compile(loss=BinaryCrossentropy())
    model_builder.model.fit(X_train, y_train, epochs=100)


if __name__ == "__main__":
    # experiment()
        # Create three random tensors with shape (3,2)
    tensor1 = tf.random.normal(shape=(3,2))
    tensor2 = tf.random.normal(shape=(3,2))
    tensor3 = tf.random.normal(shape=(3,2))

    # Print the tensors
    print("Tensor 1:\n", tensor1)
    print("Tensor 2:\n", tensor2)
    print("Tensor 3:\n", tensor3)