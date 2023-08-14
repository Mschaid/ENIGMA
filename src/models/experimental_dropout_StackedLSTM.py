import os
import pandas as pd
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Bidirectional,
                                     Dense,
                                     LSTM,
                                     Lambda,
                                     Dropout,
                                     )


class StackedLSTM(Model):
    """_summary_

    Parameters
    ----------
    Model : _type_
        _description_
    """

    def __init__(self,
                 sequence_length,
                 num_features,
                 lstm_units,
                 dropout_rate=0.2):
        super(StackedLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate


        self.input_dimensions = (self.sequence_length, self.num_features)

        # Define layers
        self.lambda_1 = Lambda(lambda x: tf.expand_dims(
            x, axis=-1), input_shape=[None], name='Lambda_1')
        
        self.lstm_1 = LSTM(self.lstm_units, input_shape=self.input_dimensions,
                           return_sequences=True, name='LSTM_1')
        self.dropout_1 = Dropout(self.dropout_rate, name='Dropout_1')
        
        self.lstm_2 = LSTM(self.lstm_units, name='LSTM_2')
        self.dropout_2 = Dropout(self.dropout_rate, name='Dropout_2')
        
        self.dense = Dense(1, name='Dense_output')

    def call(self, inputs):
        x = self.lambda_1(inputs)
        x = self.lstm_1(x)
        x = self.dropout_1(x)
        x = self.lstm_2(x)
        x = self.dropout_2(x)
        x = self.dense(x)
        return x
