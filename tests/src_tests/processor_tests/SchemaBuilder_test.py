import unittest
from sys import platform
import logging
import os

import pandas as pd
from src.data_processing.processors.SchemaBuilder import SchemaBuilder
from src.utilities.exceptions import *
from src.utilities.logger_helpers import *

current_dir = os.getcwd()

test_logs_dir = os.path.join(current_dir, 'tests', 'test_logs')
test_data_path = os.path.join(current_dir, 'data')

# if platform == 'win32':

#     test_logs_dir = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\results\logs\test_logs'
#     test_data_path = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data'

# elif platform == 'darwin':
#     test_logs_dir = test_logs_dir.replace('\\', '/')
#     test_data_path = test_data_path.replace('\\', '/')

TestLogger = set_logger_config(
    directory=test_logs_dir,
    file_name='test_logs.log')


class TestSchemaBuilder(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.sb = SchemaBuilder(test_data_path)
        self.sb.test_filetype_error_path = r'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_2.txt'
        self.sb.test_search_keys = ['test_1', 'test_2']
        self.sb.test_file_types = ['csv', 'feather', 'pkl']
        self.sb.expected_filesearch_results = ['C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\csv_test\\test_1.csv',
                                               'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\csv_test\\test_2.csv',
                                               'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\feather_test\\test_1.feather',
                                               'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\feather_test\\test_2.feather',
                                               'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_1.pkl',
                                               'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_2.pkl']

        self.sb.expected_stored_file_path = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data\test_save_schema\data_schema.pkl'

    def test_collect_data_files(self):

        self.sb.collect_data_files(
            self.sb.test_search_keys, self.sb.test_file_types)

        self.assertEqual(self.sb.expected_filesearch_results,
                         self.sb.matched_search_keys)
        self.assertEqual(len(self.sb.expected_filesearch_results),
                         len(self.sb.matched_search_keys))

    def test_aggregate_data(self):

        self.sb.collect_data_files(
            self.sb.test_search_keys, self.sb.test_file_types)
        self.sb.aggregate_data()

        # assert dataframes are a list and expected length
        self.assertIsInstance(self.sb.data_frames, list)
        self.assertEqual(6, len(self.sb.data_frames))

        # append test file type expected to raies FileTypeError and assert that it does
        self.sb.matched_search_keys.append(self.sb.test_filetype_error_path)
        self.assertRaises(FileTypeError, self.sb.aggregate_data)

        # assert that aggregate_dataframes is a pandas dataframe
        self.assertIsInstance(self.sb.aggregate_dataframes, pd.DataFrame)

    def test_save_schema(self):
        new_dir_extension = 'test_save_schema'
        self.sb.save_schema(new_dir_extension=new_dir_extension)
        expected_new_dir = os.path.join(
            self.sb.data_directory, new_dir_extension)
        expected_file_name = os.path.join(expected_new_dir, 'data_schema.pkl')

        self.assertTrue(os.path.exists(expected_new_dir))
        self.assertTrue(os.path.exists(expected_file_name))

    # TODO need to get this to work still

    @unittest.expectedFailure
    def test_load_schema():
        loaded_schema = SchemaBuilder.load_schema(
            file_path=r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data\test_save_schema\data_schema.pkl')

        # self.assertIsInstance(loaded_schema, SchemaBuilder)
        # expected_data_directory = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data'

        # self.assertIsEqual(expected_data_directory,
        #                    loaded_schema.data_directory)


if __name__ == '__main__':
    TEST_LOG_DIR = '/projects/p31961/dopamine_modeling/tests/test_logs'
    TEST_LOG_FILE = 'SchemaBuilderTest.log'
    FULL_TEST_LOG_DIR = os.path.join(TEST_LOG_DIR, TEST_LOG_FILE)

    # create logger
    logging.basicConfig(filename=FULL_TEST_LOG_DIR, 
                        level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    with open(FULL_TEST_LOG_DIR, 'w') as f:
        runner = LoggingTestRunner(verbosity=2)
        unittest.main(testRunner=runner)
