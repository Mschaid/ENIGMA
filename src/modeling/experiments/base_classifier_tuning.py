import json
import os
import pandas as pd
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from src.utilities.os_helpers import set_up_directories
from src.data_processing.processors.TrainingProcessor import TrainingProcessor
from src.models.BaseClassifier import BaseClassifier

DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'
MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
EXPERIMENT_NAME = "base_classifier_tuning"

# path to experiment directory
EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)
set_up_directories(EXPERIMENT_DIR)

data = pd.read_parquet(DATA_PATH)


classifier_processor = (TrainingProcessor(data)
                        .calculate_max_min_signal()
                        .drop_colinear_columns('action_escape')
                        .query_sensor_and_sort_trials_by_subject(sensor = 'DA')
                        .split_train_val_test_by_subject(target = 'action_avoid')
                        .save_subjects_by_category(path = EXPERIMENT_DIR)
)

space = {
    "number of layers": hp.choice('number of layers', [3, 6, 9]),
    "number of units": hp.choice('number of units', [5, 10, 15, 20,25,30]),
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
    
    #set up model
    model = BaseClassifier(number_of_layers, number_of_units, dropout_rate)
    
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
        
        
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.AUC(name='auc-roc')]
    
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = metrics)
    
    #train model
    model.fit(classifier_processor.train_x,
              classifier_processor.train_y,
              batch_size = batch_size,
              epochs = epochs,
              validation_data = (classifier_processor.val_x, classifier_processor.val_y),
              )
     
    
    evaluation = model.evaluate(classifier_processor.test_x, classifier_processor.test_y)
    
    results ={}
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
                       trials = trials)
    with open(os.path.join(EXPERIMENT_DIR, 'best_trials.json'), 'w') as f:
        json.dump(best_trials, f)
    
    return best_trials

if __name__ == "__main__":
    print("Running hyperparameter optimization")

    
    best_trials = run_trials()
    
    print(best_trials)