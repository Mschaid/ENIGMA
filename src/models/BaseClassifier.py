
import os
import pandas as pd
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Input
                                     


class BaseClassifier(Model):

    def __init__(self,
                 input_units, 
                 number_of_layers,
                 number_of_units,
                 dropout_rate):
        super(BaseClassifier, self).__init__()
        
        self.input_units = input_units
        self.number_of_layers = number_of_layers
        self.number_of_units = number_of_units
        self.dropout_rate = dropout_rate

        # define layers
        self.input_layer = Input(shape=(self.input_units,), name='Input')
        self.normalization = tf.keras.layers.Normalization()
        self.dense_layers = [Dense(self.number_of_units, activation='relu',
                                   name=f'Dense_{i}') for i in range(self.number_of_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='Dropout')
        self.output_layer = Dense(1, name='Output')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.normalization(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        self.dropout(x)
        output = self.output_layer(x)
        return output
