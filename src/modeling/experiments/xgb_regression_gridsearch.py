import json
import logging
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories
print('imports')


def set_up_logger(file_path):

    logging.basicConfig(filename=file_path,
                        filemode='w',
                        level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s - %(message)s')


def process_data(data_path, experiment_dir):
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

    return processor_pipe


def grid_search(processor, model, experiment_dir, search_space):

    # set up and run grid search
    grid = GridSearchCV(model, search_space, cv=5)
    grid.fit(processor.X_train, processor.y_train)

    # get best parameters
    best_params = grid.best_params_
    best_estimator = grid.best_estimator_
    best_estimator.fit(processor.X_dev, processor.y_dev)
    dev_prediction = best_estimator.predict(processor.X_dev)
    test_prediction = best_estimator.predict(processor.X_test)
    dev_score = mean_squared_error(processor.y_dev, dev_prediction)
    test_score = mean_squared_error(processor.y_test, test_prediction)

    scores = {
        "best_params": best_params,
        "best_dev_score": dev_score,
        "best_test_score": test_score,

    }
    for k, v in best_params.items():
        if not isinstance(v, float):
            best_params[k] = float(v)

    with open(os.path.join(experiment_dir, 'grid_search_scores.json'), 'w') as f:
        json.dump(scores, f)

    logging.info(f'Gidsearch compete, results saved in {experiment_dir}')


def main():
    # global variables
    print('starting')
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "xgb_regression_gridsearch"
    EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
    set_up_directories(EXPERIMENT_DIR)
    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')

    # set up logger and directories
    set_up_logger(LOG_FILE_PATH)

    # EXPERIMENT
    logging.info(f'Created new directories: {EXPERIMENT_DIR}')
    logging.info(f'Starting experiment: {EXPERIMENT_NAME}')
    # set up search space
    SEARCH_SPACE = {
        'n_estimators': np.arange(50, 500, 100),
        'max_depth': np.arange(3, 15, 1),
        'max_leaves': np.arange(0, 15, 1),
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'gamma': np.arange(0, 5, 0.5),
        'min_child_weight': np.arange(1, 10, 1)
    }
    print('model')
    model = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric=['rmse', 'mae'])
    logging.info('Model defined, preproessing data')
    print('preprocessing')
    processor = process_data(DATA_PATH, EXPERIMENT_DIR)
    logging.info('Data processed')
    logging.info('Starting grid search')
    print('grid search')
    grid_search(processor, model, EXPERIMENT_DIR, SEARCH_SPACE)
    logging.info(f'Grid search complete: saved at {EXPERIMENT_DIR}')


if __name__ == '__main__':
    print('main')
    main()
