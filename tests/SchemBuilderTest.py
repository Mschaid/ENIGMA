from src.utilities.logger_helpers import *
from src.SchemaBuilder import SchemaBuilder
import unittest


TestLogger = set_logger_config(
    directory='/Users/michaelschaid/GitHub/dopamine_modeling/results/logs/test_logs',
    file_name='test_logs.log')


PATH = r'/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files'

if __name__ == '__main__':
    TestLogger.info(f"test log message")
    sb = SchemaBuilder(directory=PATH)
    sb.collect_data_files(search_keys=['test_1', 'test_2'], file_types=[
                          'csv', 'feather', 'pkl'])

    TestLogger.info(f"Data directory: {sb.data_directory}")
    TestLogger.info(f"Data files: {sb.data_files}")
    TestLogger.info(f"Matched file types: {sb.matched_file_types}")
    TestLogger.info(f"Matched search keys: {sb.matched_search_keys}")
