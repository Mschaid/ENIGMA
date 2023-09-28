import unittest
from src.modeling.models.SimpleLSTM import SimpleLSTM
import tensorflow as tf

# write unit tests for SimpleLSTM


class SimpleLSTM_test(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleLSTM(sequence_length=50, num_features=1)

    def test_lambda_1(self):
        self.assertEqual(self.model.lambda_1(
            tf.zeros([50, 1])).shape, tf.TensorShape([50, 1, 1]))

    def test_lstm_1(self):
        self.assertEqual(self.model.lstm_1(
            tf.zeros([50, 1, 1])).shape, tf.TensorShape([50, 64]))

    def test_dense(self):
        self.assertEqual(self.model.dense(
            tf.zeros([50, 64])).shape, tf.TensorShape([50, 1]))

    def test_call(self):
        self.assertEqual(self.model.call(
            tf.zeros([50, 1])).shape, tf.TensorShape([50, 1]))


if __name__ == '__main__':
    unittest.main()
