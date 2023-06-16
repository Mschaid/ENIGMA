import os
import pandas as pd
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Bidirectional, 
                                     Dense, 
                                     LSTM, 
                                     Lambda 
                                     )


class StackedLSTM(Model):
    def __init__(self, 
                 sequence_length,
                 num_features,
                 lstm_1_units = 64):
        super(StackedLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_1_units = lstm_1_units
        
        self.input_dimensions = (self.sequence_length, self.num_features)

        # Define layers
        self.lambda_1 = Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None], name='Lambda_1')
        self.lstm_1 =LSTM(self.lstm_1_units, input_shape = self.input_dimensions, return_sequences=True, name='LSTM_1')
        self.lstm_2 = LSTM(self.lstm_1_units, return_sequences=True, name='LSTM_2')
        self.dense = Dense(1, activation='relu', name='Dense_output')

    def call(self, inputs):
        x = self.lambda_1(inputs)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.dense(x)
        return x

