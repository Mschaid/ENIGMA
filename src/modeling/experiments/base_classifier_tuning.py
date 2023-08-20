import json
import os
import pandas as pd
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories
from src.data_processing.processors.TrainingProcessor import TrainingProcessor
from src.models.BaseClassifier import BaseClassifier

DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
EXPERIMENT_NAME = "base_classifier_tuning"

# path to experiment directory
EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
set_up_directories(EXPERIMENT_DIR)

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


space = {
    "number of layers": hp.choice('number of layers', [1, 3, 6, 9, 18]),
    "number of units": hp.choice('number of units', [5, 10, 15, 20, 25, 30]),
    "dropout rate": hp.choice('dropout rate', [0.1, 0.2, 0.3]),
    "learning rate": hp.choice('learning rate', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    "batch size": hp.choice('batch size', [32, 64, 128, 256, 512]),
    "epochs": hp.choice('epochs', [100, 200, 300, 400, 500]),
    "optimizers": hp.choice('optimizers', ['adam', 'sgd'])

}

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

    results = {}
    results['params'] = params
    for name, value in zip(model.metrics_names, evaluation):
        results[name] = value
    # results['model'] = model
    results['status'] = STATUS_OK
    # all_results.append(results)
    with open(os.path.join(EXPERIMENT_DIR, 'results.json'), 'a+') as f:
        json.dump(results, f, indent=1)

    return results


def run_trials():
    best_trials = fmin(objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=100,
                       trials=trials)
    with open(os.path.join(EXPERIMENT_DIR, 'best_trials.json'), 'a+') as f:
        json.dump(best_trials, f)

    return best_trials


if __name__ == "__main__":
    print("Running hyperparameter optimization")

    best_trials = run_trials()

    print(best_trials)
