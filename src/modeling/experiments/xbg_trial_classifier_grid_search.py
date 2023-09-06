import json
import logging
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from xgboost_ray import RayDMatrix, RayParams, train

from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories, set_up_logger

def process_data(data_path, experiment_dir):
    processor_pipe = (ClassifierPipe(data_path)
             .read_raw_data()
             .calculate_max_min_signal()
             .drop_columns(["event", "trial"])
             .split_data(test_size=0.2,
                test_dev_size=0.5, 
                split_group = "mouse_id", 
                stratify_group = "sex", 
                target='action',
                save_subject_ids=True,
                path_to_save =os.path.dirname(experiment_dir))
            .transorm_data(numeric_target_dict={'avoid': 1, 'escape': 0})
    )
    return processor_pipe

def grid_search(processor, model, experiment_dir, search_space):
    grid = GridSearchCV(model, search_space, cv=5, scoring = 'f1')
    grid.fit(processor.X_train, processor.y_train)
    
        # get best parameters
    best_params = grid.best_params_
    best_estimator = grid.best_estimator_
    best_estimator.fit(processor.X_dev, processor.y_dev)
    dev_prediction = best_estimator.predict(processor.X_dev)
    test_prediction = best_estimator.predict(processor.X_test)
    dev_score = f1_score(processor.y_dev, dev_prediction)
    test_score = f1_score(processor.y_test, test_prediction)
    

    best_params["dev_f1_score"] =  dev_score
    best_params["test_f1_score"] =  test_score
    
    for k,v in best_params.items():
        if isinstance(v, float):
            best_params[k] = float(v)
    
        
    with open(os.path.join(experiment_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(best_params, f, indent='auto')

def main():
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "xbg_trial_classifier_ray_tuner"
    EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
    set_up_directories(EXPERIMENT_DIR)
    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')
        
    # set up logger and directories
    set_up_logger(LOG_FILE_PATH)
    
    #EXPERIMENT
    logging.info(f'Created new directories: {EXPERIMENT_DIR}')
    logging.info(f'Starting experiment: {EXPERIMENT_NAME}')
    
    
    SEARCH_SPACE = {
    "n_estimators": np.arange(50, 500, 100),
    "learning_rate": np.arange(0.01, 0.3, 0.01),
    "max_depth": np.arange(3, 15, 1),
    "min_child_weight": np.arange(1, 10, 1),
    "gamma": np.arange(0, 5, 0.5),
    "booster": ['gbtree', 'gblinear', 'dart'],
    "subsample": np.arange(0, 1, 0.2),
    "reg_lambda": np.arange(0, 5, 0.5)
    
    }
    model = xgb.XGBClassifier(objective='binary:logistic')
    logging.info('Model defined, preproessing data')
    processor = process_data(DATA_PATH, EXPERIMENT_DIR)
    logging.info('Data processed')
    logging.info('Starting grid search')
    grid_search(processor, model, EXPERIMENT_DIR, SEARCH_SPACE)
    logging.info(f'Grid search complete: saved at {EXPERIMENT_DIR}')
    


main()