import json
import os
import pandas as pd
import tensorflow as tf
import logging

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories
from src.models.BaseClassifier import BaseClassifier

DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
EXPERIMENT_NAME = "base_classifier_tuning"

# path to experiment directory
EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
set_up_directories(EXPERIMENT_DIR)

LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')
logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')

processor_pipe = (ClassifierPipe(DATA_PATH)
                  .read_raw_data()
                  .calculate_max_min_signal()
                  .split_data(test_size=0.3,
                              test_dev_size=0.5,
                              split_group="mouse_id",
                              stratify_group="sex",
                              target='action',
                              save_subject_ids=True,
                              path_to_save=os.path.dirname(EXPERIMENT_DIR))
                  .transorm_data(numeric_target_dict={'avoid': 1, 'escape': 0})
                  )
logging.info('Data processed')

space = {
    "number of layers": hp.choice('number of layers', [1, 3, 6, 9, 18]),
    "number of units": hp.choice('number of units', [5, 10, 15, 20, 25, 30]),
    "dropout rate": hp.choice('dropout rate', [0.1, 0.2, 0.3]),
    "learning rate": hp.choice('learning rate', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    "batch size": hp.choice('batch size', [32, 64, 128, 256, 512]),
    "epochs": hp.choice('epochs', [100, 200, 300, 400, 500]),
    "optimizers": hp.choice('optimizers', ['adam', 'sgd'])

}
logging.info(f"space: {space}")
trials = Trials()


def objective(params):
    # all_results=[]
    number_of_layers = params['number of layers']
    number_of_units = params['number of units']
    dropout_rate = params['dropout rate']
    learning_rate = params['learning rate']
    batch_size = params['batch size']
    epochs = params['epochs']
    optimizer = params['optimizers']

    # set up model
    model = BaseClassifier(number_of_layers, number_of_units, dropout_rate)

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc-roc')]

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=metrics)

    # train model
    model.fit(processor_pipe.X_train,
              processor_pipe.y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(processor_pipe.X_dev, processor_pipe.y_dev),
              )

    evaluation = model.evaluate(processor_pipe.X_test, processor_pipe.y_test)

    def calculate_f1_score(precission, recall):
        f1 = 2 * (precission * recall) / (precission + recall)
        return f1

    all_results = {}
    all_results['params'] = params
    for name, value in zip(model.metrics_names, evaluation):
        all_results[name] = value
    # results['model'] = model
    all_results['f1_score'] = calculate_f1_score(evaluation[1], evaluation[2])
    all_results['status'] = STATUS_OK
    evaluation.append(all_results['f1_score'])

    results_list = []
    results_list.append(all_results)
    with open(os.path.join(EXPERIMENT_DIR, 'results.json'), 'a+') as f:
        json.dump(results_list, f, indent='auto')

    f1_score = evaluation[-1]

    return -1 * f1_score


def run_trials():
    logging.info('Running hyperparameter optimization')
    best_trials = fmin(objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=400,
                       trials=trials)

    for k, v in best_trials.items():
        best_trials[k] = float(v)

    best_trials_list = []
    best_trials_list.append(best_trials)
    with open(os.path.join(EXPERIMENT_DIR, 'best_trial.json'), 'a+') as f:
        json.dump(best_trials_list, f, indent='auto')

    return best_trials


if __name__ == "__main__":
    logging.info('Running hyperparameter optimization')
    best_trials = run_trials()
    logging.info('Hyperparameter optimization complete')
    logging.info(f'Best trials: {best_trials}')
