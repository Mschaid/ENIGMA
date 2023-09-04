import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories, set_up_logger


    
def process_data(data_path, experiment_dir):
    processor_pipe = (ClassifierPipe(data_path)
                    .read_raw_data()
                    .calculate_max_min_signal()
                    .split_data(test_size=0.3,
                                test_dev_size=0.5,
                                split_group="mouse_id",
                                stratify_group="sex",
                                target='action',
                                save_subject_ids=True,
                                path_to_save=os.path.dirname(experiment_dir))
                    .transorm_data(numeric_target_dict={'avoid': 1, 'escape': 0})
                    )

    return processor_pipe



def main():
    # global variables
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "xgb_regression_gridsearch"
    EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log') 
    
    # set up logger and directories
    set_up_logger(LOG_FILE_PATH)
    set_up_directories(EXPERIMENT_DIR)
    
    #EXPERIMENT
    logging.info(f'Created new directories: {EXPERIMENT_DIR}')
    logging.info(f'Starting experiment: {EXPERIMENT_NAME}')
    # set up search space

    

    logging.info('Model defined, preproessing data')
    processor = process_data(DATA_PATH, EXPERIMENT_DIR)
    
    
    logging.info('Data processed')
    logging.info('Starting grid search')


    
if __name__=='__main__':
    main()