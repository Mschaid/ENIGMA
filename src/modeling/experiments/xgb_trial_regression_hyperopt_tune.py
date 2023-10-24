import json 
import logging
import numpy as np
import os
import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
#local imports
from src.utilities.os_helpers import set_up_directories, set_up_logger
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories, set_up_logger


def process_data(data_path, experiment_dir):
    logging.info('Processing data')
    processor_pipe = (ClassifierPipe(data_path)
                      .read_raw_data()
                      .calculate_max_min_signal()
                      .calculate_percent_avoid()
                      .drop_features(["event", "action", "trial", "trial_count", "num_avoids", "max_trial"])
                      .split_data(test_size=0.3,
                                  test_dev_size=0.5,
                                  split_group="mouse_id",
                                  stratify_group="sex",
                                  target='ratio_avoid',
                                  save_subject_ids=True,
                                  path_to_save=experiment_dir)
                      .transform_data()
                      )
    logging.info('Data processed')
    return processor_pipe

def hyperopt_experiment(processor, space, max_evals):
    logging.info('Running hyperopt')
    def objective(params):
        model = xgb.XGBClassifier(objective='binary:logistic', **params)
        predictions = cross_val_predict(model, processor.X_train, processor.y_train, cv=5)
        score = f1_score(processor.y_train, predictions)
        return {'loss': -score, 'status': STATUS_OK}
    
    trils = Trials()
    best = fmin(fn=objective, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=max_evals,
                trials = trials)
    logging.info('Hyperopt complete')
    logging.info(f'Best params: {space_eval(space, best)}')

def main():
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "xgb_regression_hyperopt"
    EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
    set_up_directories(EXPERIMENT_DIR)
    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')

    # set up logger and directories
    set_up_logger(LOG_FILE_PATH)

    PROCESSOR = process_data(DATA_PATH, EXPERIMENT_DIR)

    SEARCH_SPACE = {
        "n_estimators": hp.uniform('n_estimators', 50, 500),
        "learning_rate": hp.uniform('learning_rate', 0.01, 0.2),
        "max_depth": hp.uniform('max_depth', 3, 15),
        "min_child_weight": hp.uniform('min_child_weight', 1, 10),
        "gamma": hp.uniform('gamma', 0, 5),
        "booster": hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        "subsample": hp.uniform('subsample', 0, 1),
        "reg_lambda": hp.uniform('reg_lambda', 0, 5)
    }

    hyperopt_experiment(processor=PROCESSOR, space=SEARCH_SPACE, max_evals=100)

if __name__ == "__main__":
    logging.info('Starting experiment')
    main()
    logging.info('Experiment complete')

    