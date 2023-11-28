import json
import logging
import numpy as np
import os
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# local imports
from src.utilities.os_helpers import set_up_directories, set_up_logger
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe


def process_data(data_path, experiment_dir):
    logging.info('Processing data')
    processor_pipe = (ClassifierPipe(data_path)
                      .read_raw_data()
                      .calculate_max_min_signal()
                      .calculate_percent_avoid()
                      .drop_features(["event", "action", "trial", "trial_count", "num_avoids", "max_trial"])
                      .stratify_and_split_by_mouse(test_size=0.3,
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
        model = xgb.XGBRegressor(
            objective='reg:squarederror', eval_metric=['rmse', 'mae'], **params)
        model.fit(processor.X_train, processor.y_train)
        scores = -cross_val_score(model, processor.X_dev,
                                  processor.y_dev, cv=5)
        mean_score = np.mean(scores)
        return {'loss': mean_score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    # best_trials_path = os.path.join(experiment_dir, 'best_trial.json')
    # with open(best_trials_path, 'w') as f:
    #     json.dump(best, f, indent=4)

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
        "n_estimators": hp.choice('n_estimators', [50, 100, 150, 200, 250]),
        "learning_rate": hp.choice('learning_rate', np.arange(0.05, 0.2, 0.5)),
        "max_depth": hp.choice('max_depth', np.arange(3, 15, 3)),
        "min_child_weight": hp.choice('min_child_weight', np.arange(1, 10, 1)),
        "gamma": hp.choice('gamma', np.arange(0, 5, 1)),
        "booster": hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        "subsample": hp.choice('subsample', np.arange(0, 1, 0.2)),
        "reg_lambda": hp.choice('reg_lambda', np.arange(0, 5, 0.5))
    }

    hyperopt_experiment(processor=PROCESSOR,
                        space=SEARCH_SPACE,
                        max_evals=1000)


main()
