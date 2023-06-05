import logging

from src.processors.Preprocessor import Preprocessor


# Set up logging
logging.basicConfig(filename = '/projects/p31961/dopamine_modeling/results/logs/processing_logs/process_gaby_data.log',
                    level = logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s - %(message)s')

PATH_TO_DATA = r'/projects/p31961/gaby_data/aggregated_data/aggregated_data.parquet.gzp'
PATH_TO_SAVE = r'/projects/p31961/gaby_data/aggregated_data/data_pipeline'
gaby_processor = Preprocessor(
    processor_name = '5_day_training_gaby',
    path_to_data = PATH_TO_DATA,
    path_to_save = PATH_TO_SAVE,
    features = ['day', 'time', 'trial','event_cue', 
                'event_shock','sensor_D1','sensor_D2',
                'sensor_DA'],
    target = 'signal')
    
def process_and_store_data():
    logging.info(f'loading data from {gaby_processor.path_to_data}')
    gaby_processor.load_data()
    logging.info('one hot encoding')
    gaby_processor.one_hot_encode(labels = ['event', 'sensor'])
    logging.info('splitting data by query')
    gaby_processor.split_train_by_query('day', 5, processed_data=True)
    logging.info('downsampling data')
    gaby_processor.downsample_train_and_test_datasets(n = 100)
    logging.info('saving datasets')
    gaby_processor.save_datasets_to_parquet()
    # gaby_processor.save_data_to_h5()
    # logging.info('saving processor')
    logging.info('data saved')
    # gaby_processor.save_processor()
    # logging.info(f'processor saved to {gaby_processor.path_to_save_processor}')
    
if __name__ == '__main__':
    # process_and_store_processor()
    logging.info('initiating processing and storing processor')
    process_and_store_data()
    logging.info('done processing and storing processor')