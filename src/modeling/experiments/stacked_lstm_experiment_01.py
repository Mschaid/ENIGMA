import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.modeling.Models.SimpleLSTM import SimpleLSTM
from src.modeling.Models.StackedLSTM import StackedLSTM
from src.utilities.tensorflow_helpers import set_tensorboard


def stacked_lstm_experiment_01(units):
    

    dataset_dir = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_downsampled/datasets'
    X_train_path = os.path.join(dataset_dir, '5_day_training_gaby_downsampled_X_train.parquet.gzip')
    X_test_path = os.path.join(dataset_dir, '5_day_training_gaby_downsampled_X_test.parquet.gzip')
    y_train_path = os.path.join(dataset_dir, '5_day_training_gaby_downsampled_y_train.parquet.gzip')
    y_test_path = os.path.join(dataset_dir, '5_day_training_gaby_downsampled_y_test.parquet.gzip')

    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)
    y_train = pd.read_parquet(y_train_path)
    y_test = pd.read_parquet(y_test_path)
    
    save_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/model_experimentation/stacked_lstm_experiment_01"
    
    for unit in units:
        
        model_name = f'StackedLSTM_{unit}_units'
        tensorboard_callback = set_tensorboard(model_name, save_dir)
        
        model = StackedLSTM(sequence_length = 90,
                           num_features = X_train.shape[1],
                           lstm_1_units=units
                           )
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
        model.fit(X_train, y_train, epochs = 10,  callbacks = [tensorboard_callback])
        model.evaluate(X_test, y_test, callbacks = [tensorboard_callback])
        
if __name__ == "__main__":
    units = range(30, 150, 30)
    stacked_lstm_experiment_01(units)