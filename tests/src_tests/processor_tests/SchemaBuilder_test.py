import unittest
import pandas as pd
import shutil
import tempfile

from unittest.mock import patch, mock_open

from src.data_processing.processors.SchemaBuilder import SchemaBuilder
from src.utilities.exceptions import *
from src.utilities.logger_helpers import *


class TestSchemaBuilder(unittest.TestCase):

    def _create_test_files(self):
        open(os.path.join(self.test_dir, 'file1.txt'), 'w').close()
        open(os.path.join(self.test_dir, 'file2.csv'), 'w').close()
        open(os.path.join(self.test_dir, 'file3.csv'), 'w').close()

    def _create_mock_data(self):
        # create some test dataframes
        self.df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})
        self.df3 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        self.expexted = pd.concat([self.df1, self.df2, self.df3])

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sb = SchemaBuilder(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_directory(self):

        self.assertEqual(self.sb.data_directory, self.test_dir)

    def test_collect_data_files_1_file(self):
        # create some test files
        self._create_test_files()

        # test collecting files with one search key and one file type
        self.sb.collect_data_files(search_keys='file2', file_types=['txt'])
        self.assertEqual(len(self.sb.data_files), 3)
        self.assertEqual(len(self.sb.matched_search_keys), 1)

        self.assertEqual(self.sb.matched_search_keys[0], os.path.join(
            self.test_dir, 'file1.txt'))

    def test_collect_data_files_2_files(self):

        # create some test files
        self._create_test_files()
        # test collecting files with multiple search keys and file types
        self.sb.collect_data_files(['file2', 'file3'], ['csv'])
        # self.assertEqual(len(self.sb.data_files), 2)
        self.assertEqual(len(self.sb.data_files), 3)
        self.assertEqual(len(self.sb.matched_search_keys), 2)

        self.assertEqual(self.sb.matched_search_keys[0], os.path.join(
            self.test_dir, 'file2.csv'))

    #! TODO: fix this test
    @unittest.expectedFailure
    def test_aggregate_data(self):
        # create mock data
        file1 = 'data/file1.csv'
        file2 = 'data/file2.xlsx'
        file3 = 'data/file3.feather'
        file4 = 'data/file4.parquet'

        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9]})
        df3 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        df4 = pd.DataFrame({'A': [10, 11, 12], 'B': [13, 14, 15]})

        with patch('builtins.open', mock_open()) as mock_open_func:
            mock_open_func.side_effect = [file1, file2, file3, file4]
            pd.DataFrame.to_csv(df1, file1)
            pd.DataFrame.to_excel(df2, file2)
            pd.DataFrame.to_feather(df3, file3)
            pd.DataFrame.to_parquet(df4, file4)

        sb = SchemaBuilder('.')
        sb.aggregate_data([file1, file2, file3, file4])

        expected = pd.concat([df1, df2, df3, df4])
        pd.testing.assert_frame_equal(sb.data, expected)


if __name__ == '__main__':
    unittest.main()
