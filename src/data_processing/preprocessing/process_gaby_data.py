import logging

from src.data_processing.processors.Preprocessor import Preprocessor
from src.utilities.gaby_processing_helpers import merge_latency_data, merge_sex_data, full_data_feature_extraction, assign_cumulative_trials


# Set up logging
LOG_FILE_PATH = '/projects/p31961/ENIGMA/results/logs/processing_logs/process_gaby_data.log'

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s - %(message)s')


PATH_TO_DATA = r'/projects/p31961/gaby_data/aggregated_data/downsampled_aggregated_data.parquet.gzp'
PATH_TO_LATENCY_DATA = "/projects/p31961/gaby_all_raw_data/AA_Latencies.xlsx"
PATH_TO_SEX_DATA = "/projects/p31961/gaby_all_raw_data/AA_ListofSex.xlsx"


PATH_TO_SAVE = r'/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset'
gaby_processor = Preprocessor(
    processor_name='5_day_training_full_dataset',
    path_to_data=PATH_TO_DATA,
    path_to_save=PATH_TO_SAVE,
    features=['time', 'mouse_id', 'sex_M', 'day', 'trial', 'learning_phase',
              'event_cue', 'event_escape', 'event_avoid',
              'latency', 'event_shock', 'sensor_D1', 'sensor_D2', 'sensor_DA'],
    target='signal'
)


def process_and_store_data():

    logging.info(f'loading data from {gaby_processor.path_to_data}')
    gaby_processor.load_data()

    logging.info('merging latency data')
    gaby_processor.data = merge_latency_data(
        gaby_processor.data, PATH_TO_LATENCY_DATA)
    logging.info('merging sex data')

    gaby_processor.data = merge_sex_data(gaby_processor.data, PATH_TO_SEX_DATA)

    logging.info('extracting features')
    gaby_processor.data = full_data_feature_extraction(gaby_processor.data)

    logging.info("assigning cumulative trials")
    gaby_processor.data = assign_cumulative_trials(gaby_processor.data)

    logging.info('one hot encoding')
    gaby_processor.one_hot_encode(labels=['event', 'sensor', 'sex'])

    logging.info('splitting data by query')
    gaby_processor.split_train_by_query('day', 5, processed_data=True)

    logging.info('saving data')
    gaby_processor.save_datasets_to_parquet(save_downsampled=False)
    logging.info('data saved')


if __name__ == '__main__':

    logging.info('initiating processing and storing processor')
    process_and_store_data()
    logging.info('done processing and storing processor')
