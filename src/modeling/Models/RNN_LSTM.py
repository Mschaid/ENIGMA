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


class SimpleLSTM(Model):
    def __init__(self, sequence_length, input_dimensions):
        super().__init__()
        
        self.lstm_input_shape = (sequence_length, input_dimensions)
        
        self.lambda_1 = Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None], name='Lambda_1')
        self.lstm_1 =LSTM(32, input_shape = self.lstm_input_shape, return_sequences=True, name='LSTM_1')

        self.dense = Dense(1, activation='relu', name='Dense_output')

    def call(self, inputs):
        x = self.lambda_1(inputs)
        x = self.lstm_1(x)
        return self.dense(x)
