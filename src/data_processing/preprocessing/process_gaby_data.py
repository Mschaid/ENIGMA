import logging

from src.data_processing.processors.Preprocessor import Preprocessor
from src.utilities.gaby_processing_helpers import (merge_latency_data,
                                                   merge_sex_data,
                                                   merge_behavior_data,
                                                   full_data_feature_extraction,
                                                   assign_cumulative_trials)


# Set up logging
LOG_FILE_PATH = '/projects/p31961/ENIGMA/results/logs/processing_logs/process_gaby_data.log'

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s - %(message)s')


PATH_TO_DATA = r'/projects/p31961/gaby_data/aggregated_data/downsampled_aggregated_data.parquet.gzp'
PATH_TO_LATENCY_DATA = "/projects/p31961/gaby_all_raw_data/AA_Latencies.xlsx"
PATH_TO_SEX_DATA = "/projects/p31961/gaby_all_raw_data/AA_ListofSex.xlsx"
PATH_TO_BEHAVIORAL_DATA = "/projects/p31961/gaby_all_raw_data/BehaviorData_ActiveAvoidance.xlsx"


PATH_TO_SAVE = r'/projects/p31961/gaby_data/aggregated_data/raw_data'
gaby_preprocessor = Preprocessor(
    processor_name='raw_data',
    path_to_data=PATH_TO_DATA,
    path_to_save=PATH_TO_SAVE
)


def process_and_store_data():

    logging.info(f'loading data from {gaby_preprocessor.path_to_data}')
    gaby_preprocessor.load_data()

    logging.info('merging behavioral data')
    gaby_preprocessor.data = merge_behavior_data(
        gaby_preprocessor.data, PATH_TO_BEHAVIORAL_DATA)
    logging.info('merging latency data')
    gaby_preprocessor.data = merge_latency_data(
        gaby_preprocessor.data, PATH_TO_LATENCY_DATA)

    logging.info('merging sex data')
    gaby_preprocessor.data = merge_sex_data(
        gaby_preprocessor.data, PATH_TO_SEX_DATA)

    logging.info('extracting features')
    gaby_preprocessor.data = full_data_feature_extraction(
        gaby_preprocessor.data)

    logging.info("assigning cumulative trials")
    gaby_preprocessor.data = assign_cumulative_trials(gaby_preprocessor.data)

    logging.info('saving data')
    gaby_preprocessor.save_datasets_to_parquet(save_downsampled=False)
    logging.info('data saved')


if __name__ == '__main__':

    logging.info('initiating processing and storing processor')
    process_and_store_data()
    logging.info('done processing and storing processor')
