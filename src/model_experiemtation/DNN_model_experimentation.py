
"""

initial experimentation with DNN model using LSTM 
"""

from src.processors.Preprocessor import Preprocessor
from src.model_experiemtation.ModelExperimentor import TFModelExperimentor

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split
import logging
from time import sleep
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, LSTM
from tensorflow.keras import Sequential

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError


if __name__ == '__main__':
    X_train = pd.read_pickle('/projects/p31961/dopamine_modeling/data/gaby_test/X_train.pkl')
    X_test = pd.read_pickle('/projects/p31961/dopamine_modeling/data/gaby_test/X_test.pkl')
    y_train = pd.read_pickle('/projects/p31961/dopamine_modeling/data/gaby_test/y_train.pkl')
    y_test = pd.read_pickle('/projects/p31961/dopamine_modeling/data/gaby_test/y_test.pkl')
    
    print('data loaded')


    sequential_model = Sequential(
        [
            LayerNormalization(axis=-1, name="normalization"),
            LSTM(64, name="lstm_1"),
            Dense(128, activation='relu', name="dense_1"),
            Dense(1, activation='relu', name="dense_2")
        ]
    )



    # model_kwargs = {"optimizer": "adam",
    #                 "loss": "binary_crossentropy",
    #                 "metrics": ["accuracy"]}
    logs_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir, histogram_freq=1)

    sequential_model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mse, accuracy"])
    sequential_model.fit(X_train, y_train, epochs=200,
                     batch_size=30, callbacks=[tensorboard_callback])
    prediction = sequential_model.predict(X_test)
    sequential_model.evaluate(X_test, y_test)
    
    

    path_to_save = '/projects/p31961/dopamine_modeling/results/models'
    model_name = 'DNN_test_model'
    sequential_model.save(os.path.join(path_to_save, model_name))
    

    print(sequential_model.summary())
    print(prediction.shape)