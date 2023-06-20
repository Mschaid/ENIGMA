import unittest
import numpy as np
from src.modeling.Models.StackedLSTM import StackedLSTM

class TestStackedLSTM(unittest.TestCase):
    def setUp(self):
        self.sequence_length = 10
        self.num_features = 5
        self.lstm_1_units = 64
        self.model = StackedLSTM(self.sequence_length, self.num_features, self.lstm_1_units)

    def test_input_shape(self):
        inputs = np.zeros((1, self.sequence_length, self.num_features))
        output = self.model(inputs)
        self.assertEqual(output.shape, (1, 1))

    def test_output_range(self):
        inputs = np.random.rand(1, self.sequence_length, self.num_features)
        output = self.model(inputs)
        self.assertTrue(np.all(output >= 0))

if __name__ == '__main__':
    unittest.main()