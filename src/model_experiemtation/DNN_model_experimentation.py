
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
from tensorflow.keras.layers import Dense, Normalization, LSTM
from tensorflow.keras import Sequential

from tensorflow.keras.losses import BinaryCrossentropy


if __name__ == '__main__':
    PATH = '/projects/p31961/gaby_data/aggregated_data/processors/5_day_training_gaby.pkl'
    proc = Preprocessor().load_processor(PATH)
    #normalize training data
    X_train = proc.X_train.sample(1000)
    y_train = proc.y_train.sample(1000)
    X_test = proc.X_test.sample(1000)
    y_test = proc.y_test.sample(1000)
    

    print('proc loaded')

    sequential_model = Sequential(
        [
            Normalization(axis=-1, name="normalization"),
            # LSTM()
            Dense(128, activation='relu', name="dense_1"),
            Dense(128, activation='relu', name="dense_2"),
            Dense(128, activation='relu', name="dense_3")
        ]
    )

    tf_seq = TFModelExperimentor(sequential_model, preprocessor=proc)

    model_kwargs = {"optimizer": "adam",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"]}
    logs_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir, histogram_freq=1)

    tf_seq.model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    tf_seq.model.fit(X_train, y_train, epochs=200,
                     batch_size=30, callbacks=[tensorboard_callback])
    prediction = tf_seq.model.predict(X_test)
    tf_seq.model.evaluate(X_test, y_test)
    
    

    path_to_save = '/projects/p31961/dopamine_modeling/results/models'
    model_name = 'DNN_test_model'
    tf_seq.model.save(os.path.join(path_to_save, model_name))
    

    print(tf_seq.model.summary())
