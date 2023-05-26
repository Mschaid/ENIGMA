import unittest
from unittest.mock import patch
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber

from src.processors.Preprocessor import Preprocessor
from src.model_experimentation.model_experimentation import (
    read_data,
    split_data_by_trial,
    build_model,
    set_tensorboard,
    train_model,
    evaluate_model,
    inference,
    save_model
)

class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'trial': [1, 2, 3, 4, 5],
            'signal': [0.1, 0.2, 0.3, 0.4, 0.5],
        })

    def tearDown(self):
        pass

    def test_read_data(self):
        with patch('pd.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = self.df
            data = read_data('path/to/data.parquet')
            mock_read_parquet.assert_called_once_with('path/to/data.parquet')
            pd.testing.assert_frame_equal(data, self.df)

    def test_split_data_by_trial(self):
        X_train, y_train, X_test, y_test = split_data_by_trial(self.df, 3)
        expected_X_train = pd.DataFrame({
            'trial': [1, 2, 3],
            'signal': [0.1, 0.2, 0.3],
        })
        expected_y_train = pd.Series([0.1, 0.2, 0.3])
        expected_X_test = pd.DataFrame({
            'trial': [4, 5],
            'signal': [0.4, 0.5],
        })
        expected_y_test = pd.Series([0.4, 0.5])

        pd.testing.assert_frame_equal(X_train, expected_X_train)
        pd.testing.assert_series_equal(y_train, expected_y_train)
        pd.testing.assert_frame_equal(X_test, expected_X_test)
        pd.testing.assert_series_equal(y_test, expected_y_test)

    def test_build_model(self):
        model = build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 2)
        self.assertIsInstance(model.layers[0], Dense)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertEqual(model.layers[0].units, 128)
        self.assertEqual(model.layers[1].units, 1)

    def test_set_tensorboard(self):
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20210526-120000"
            callback = set_tensorboard('sequential_prototype')
            expected_log_dir = "/projects/p31961/dopamine_modeling/results/logs/training_logs/sequential_prototype/20210526-120000"
            self.assertIsInstance(callback, tf.keras.callbacks.TensorBoard)
            self.assertEqual(callback.log_dir, expected_log_dir)

    def test_train_model(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])

        with patch.object(model, 'compile') as mock_compile, \
                patch.object(model, 'fit') as mock_fit:
            tensorboard_callback = tf.keras.callbacks.TensorBoard()
            train_model(model, X_train, y_train, tensorboard_callback)
            mock_compile.assert_called_once()
            mock_fit.assert_called_once_with(
                X_train, y_train, batch_size=30, epochs=100,
                callbacks=[tensorboard_callback]
            )

    def test_evaluate_model(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        X_test = np.array([[4], [5]])
        y_test = np.array([8, 10])

        with patch.object(model, 'evaluate') as mock_evaluate:
            evaluate_model(model, X_test, y_test)
            mock_evaluate.assert_called_once_with(X_test, y_test)

    def test_inference(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        X_test = np.array([[4], [5]])

        with patch.object(model, 'predict') as mock_predict:
            inference(model, X_test)
            mock_predict.assert_called_once_with(X_test)

    def test_save_model(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        model_id = 'my_model'
        path_to_save = '/path/to/save'

        with patch.object(model, 'save') as mock_save, \
                patch.object(os.path, 'join') as mock_join:
            save_model(model, path_to_save, model_id)
            mock_join.assert_called_once_with(path_to_save, model_id)
            mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
