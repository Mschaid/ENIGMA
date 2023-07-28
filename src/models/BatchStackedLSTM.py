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


class BatchStackedLSTM(Model):
    """_summary_

    Parameters
    ----------
    Model : _type_
        _description_
    """

    def __init__(self,
                 input_dimensions,
                 lstm_units):
        super(BatchStackedLSTM, self).__init__()
        self.input_dimensions = input_dimensions
        self.lstm_units = lstm_units

        # Define layers
        self.lstm_1 = LSTM(self.lstm_units, input_shape=self.input_dimensions,
                           return_sequences=True, name='LSTM_1')
        self.lstm_2 = LSTM(self.lstm_units, name='LSTM_2')
        self.dense = Dense(1, name='Dense_output')

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.dense(x)
        return x
