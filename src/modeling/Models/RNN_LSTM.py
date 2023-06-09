import os
import pandas as pd
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda


class SimpleLSTM(Model):
    def __init__(self):
        super().__init__()
        self.lambda_1 = Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None], name = 'Lambda_1')
        self.lstm = LSTM(64, input_shape = (None, 1), return_sequences=False, name = 'LSTM')
        self.dense = Dense(1, activation='relu', name = 'Dense_output')
        
    def call(self, inputs):
        x = self.lambda_1(inputs)
        x = self.lstm(x)
        return self.dense(x)
    
if __name__=='__main__':
    model = SimpleLSTM()
    model.build(input_shape=(None, 1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()