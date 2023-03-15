from sys import platform
from src.utilities.logger_helpers import *
from src.utilities.exceptions import *
from src.SchemaBuilder import SchemaBuilder
import unittest


if platform == 'win32':

    test_logs_dir = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\results\logs\test_logs'
    test_data_path = r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data'

elif platform == 'darwin':
    test_logs_dir = test_logs_dir.replace('\\', '/')
    test_data_path = test_data_path.replace('\\', '/')

TestLogger = set_logger_config(
    directory=test_logs_dir,
    file_name='test_logs.log')


class TestSchemaBuilder(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.sb = SchemaBuilder(test_data_path)

    def test_collect_data_files(self):
        test_search_keys = ['test_1', 'test_2']
        test_file_types = ['csv', 'feather', 'pkl']
        self.sb.collect_data_files(test_search_keys, test_file_types)
        expected_results = ['C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\csv_test\\test_1.csv',
                            'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\csv_test\\test_2.csv',
                            'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\feather_test\\test_1.feather',
                            'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\feather_test\\test_2.feather',
                            'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_1.pkl',
                            'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_2.pkl']

        self.assertEqual(expected_results, self.sb.matched_search_keys)
        self.assertEqual(len(expected_results),
                         len(self.sb.matched_search_keys))

    def test_aggregate_data(self):
        test_search_keys = ['test_1', 'test_2']
        test_file_types = ['csv', 'feather', 'pkl']
        # cls.sb.collect_data_files(test_search_keys, test_file_types)
        self.sb.collect_data_files(test_search_keys, test_file_types)
        self.sb.aggregate_data()
        self.assertIsInstance(self.sb.data_frames, list)
        # print(self.sb.data_frames)
        self.assertEqual(6, len(self.sb.data_frames))
        self.assertRaises(FileTypeError, self.sb.aggregate_data,
                          'C:\\Users\\mds8301\\Documents\\Github\\dopamine_modeling\\data\\test_data_files\\pkl_test\\test_2.txt')


if __name__ == '__main__':
    unittest.main(verbosity=2)
# if __name__ == '__main__':
#     TestLogger.info(f"test log message")
#     sb = SchemaBuilder(directory=test_data_path)
#     sb.collect_data_files(search_keys=['test_1', 'test_2'], file_types=[
#                           'csv', 'feather', 'pkl'])

#     TestLogger.info(f"Data directory: {sb.data_directory}")
#     TestLogger.info(f"Data files: {sb.data_files}")
#     TestLogger.info(f"Matched file types: {sb.matched_file_types}")
#     TestLogger.info(f"Matched search keys: {sb.matched_search_keys}")
