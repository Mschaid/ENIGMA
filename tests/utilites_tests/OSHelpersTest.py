import os
import unittest
import pandas as pd
import shutil
from src.utilities.os_helpers import *


class TestOSHelpers(unittest.TestCase):
    

    def test_create_dir_with_existing_directory(self):
        # Create a temporary directory for testing
        test_dir = "/projects/p31961/dopamine_modeling/tests/utilites_tests/test_dir"
        create_dir(test_dir)

        # Test when directory already exists
        result = create_dir(test_dir)
        self.assertEqual(result, test_dir)

        # Clean up the temporary directory

        shutil.rmtree(test_dir)

    def test_create_dir_with_non_existing_directory(self):
        # Test when directory is created
        test_dir = "/projects/p31961/dopamine_modeling/tests/utilites_tests/test_dir"
        new_dir = "new_directory"
        create_dir(test_dir)
        result = create_dir(new_dir)
        self.assertEqual(result, new_dir)
        self.assertTrue(os.path.exists(new_dir))

        # Clean up the created directory
        shutil.rmtree(test_dir)
        shutil.rmtree(new_dir)
        

    def test_create_dir_with_none_path(self):
        # Test when path is None
        result = create_dir(path = None)
        self.assertIsNone(result)
        

    def test_create_new_directory(self):
        # Test creating a new directory
        file_path = "/projects/p31961/dopamine_modeling/tests/utilites_tests/test_dir"
        new_dir_extension = "new_directory"
        create_dir(file_path)
        new_directory = create_new_directoy(file_path, new_dir_extension)
        expected_directory = os.path.join(file_path, new_dir_extension)
        self.assertEqual(new_directory, expected_directory)
        self.assertTrue(os.path.exists(expected_directory))

        # Clean up the created directory
        shutil.rmtree(file_path)
        

    def test_save_dataframes_to_parquet(self):
        # Test saving DataFrames to Parquet files
        X_train = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        X_test = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
        y_train = pd.Series([1, 2, 3], name='E')
        y_test = pd.Series([4, 5, 6], name='F')

        file_path = "/projects/p31961/dopamine_modeling/tests/utilites_tests/test_dir"
        create_dir(file_path)
        save_dataframes_to_parquet(
            ('X_train', X_train),
            ('X_test', X_test),
            ('y_train', y_train),
            ('y_test', y_test),
            path_to_save=file_path
        )

        # Check if the Parquet files are saved
        self.assertTrue(os.path.exists(os.path.join(
            file_path, 'X_train.parquet.gzip')))
        self.assertTrue(os.path.exists(os.path.join(
            file_path, 'X_test.parquet.gzip')))
        self.assertTrue(os.path.exists(os.path.join(
            file_path, 'y_train.parquet.gzip')))
        self.assertTrue(os.path.exists(os.path.join(
            file_path, 'y_test.parquet.gzip')))

        # # Clean up the created Parquet files
        shutil.rmtree(file_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
