import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Lambda
from src.models.StackedLSTM import StackedLSTM


class TestStackedLSTM(unittest.TestCase):
    def setUp(self):
        self.sequence_length = 10
        self.num_features = 5
        self.lstm_units = 32
        self.model = StackedLSTM(self.sequence_length, self.num_features, self.lstm_units)

    def test_model_output_shape(self):
        input_data = np.random.rand(1, self.sequence_length, self.num_features)
        expected_output_shape = (1, 1)
        output = self.model(input_data)
        self.assertEqual(output.shape, expected_output_shape)

    def test_model_layers(self):
        expected_layers = [
            Lambda,
            LSTM,
            LSTM,
            Dense
        ]
        layers = [layer.__class__ for layer in self.model.layers]
        self.assertEqual(layers, expected_layers)

    def test_model_summary(self):
        expected_summary = 'Model: "StackedLSTM"\n' \
                           '_________________________________________________________________\n' \
                           'Layer (type)                 Output Shape              Param #   \n' \
                           '=================================================================\n' \
                           'Lambda_1 (Lambda)            (None, None, 5, 1)        0         \n' \
                           '_________________________________________________________________\n' \
                           'LSTM_1 (LSTM)                (None, None, 32)          4352      \n' \
                           '_________________________________________________________________\n' \
                           'LSTM_2 (LSTM)                (None, 32)                8320      \n' \
                           '_________________________________________________________________\n' \
                           'Dense_output (Dense)         (None, 1)                 33        \n' \
                           '=================================================================\n' \
                           'Total params: 12,705\n' \
                           'Trainable params: 12,705\n' \
                           'Non-trainable params: 0\n' \
                           '_________________________________________________________________\n'
        self.assertEqual(self.model.summary(), expected_summary)


if __name__ == '__main__':
    unittest.main()