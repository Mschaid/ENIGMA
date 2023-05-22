
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy



if __name__ == '__main__':
    PATH = '/projects/p31961/gaby_data/aggregated_data/processors/5_day_training_gaby.pkl'
    proc = Preprocessor().load_processor(PATH)
    
    sleep(60)
    print('proc loaded')

    sequential_model = Sequential (
        [
            Dense(128, activation = 'relu', name = "dense_1", input_shape = proc.X_train.columns.shape),
            Dense(128, activation = 'relu', name = "dense_2"),
            Dense(128, activation = 'relu', name = "dense_3")        
        ]
    )



    tf_seq = TFModelExperimentor(sequential_model, preprocessor = proc)
    
    model_kwargs = {"optimizer": "adam", 
                   "loss": "binary_crossentropy",
                   "metrics": ["accuracy"]}
    logs_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs_dir, histogram_freq = 1)

    
    print('downsampling')
    X_train = tf_seq.preprocessor.X_train.sample(1000)
    y_train = tf_seq.preprocessor.y_train.sample(1000)
    print('compiling model and training')
    tf_seq.model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    tf_seq.model.fit(X_train, y_train, epochs = 10, batch_size = 10, callbacks = [tensorboard_callback])
    # tf_seq.compile_model(**model_kwargs)
    # tf_seq.train_model
    
    print(tf_seq.model.summary())