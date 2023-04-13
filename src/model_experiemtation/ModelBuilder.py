
from src.processors.Preprocessor import Preprocessor
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy


PATH_TO_PROCESSED_DATA = input('Enter path to processed data: ')


class ModelBuilder:

    def __init__(self, path_to_processed_data, features=None, target=None):
        self.path_to_processed_data = path_to_processed_data
        self.features = features
        self.target = target
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.processed_data = pd.read_parquet(self.path_to_processed_data)

    def split_train_test(self, X_labels, y_label, test_size=0.2, random_state=42):
        self.features = self.processed_data[X_labels]
        self.target = self.processed_data[y_label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state)
        return self

    def split_train_by_query(self, coloumn_name, cut_off):
        self.under_cut_off_data = self.processed_data.query(
            "@column_name <= cutoff")
        self.over_cut_off_data = self.processed_data.query(
            "@column_name > cutoff")
        self.X_train = self.under_cut_off_data[self.features]
        self.y_train = self.under_cut_off_data[self.target]
        self.X_test = self.over_cut_off_data[self.features]
        self.y_test = self.over_cut_off_data[self.target]

    def save_model(self, model, model_name, path):
        model.save(os.path.join(path, model_name))

    def load_model(self, model_name, path):
        self.model_name = tf.keras.models.load_model(
            os.path.join(path, model_name))
        return self.model_name
