import os
import pandas as pd
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from src.models.StackedLSTM import StackedLSTM

dataset_dir = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_downsampled/datasets'
X_train_path = os.path.join(
    dataset_dir, '5_day_training_gaby_downsampled_X_train.parquet.gzip')
X_test_path = os.path.join(
    dataset_dir, '5_day_training_gaby_downsampled_X_test.parquet.gzip')
y_train_path = os.path.join(
    dataset_dir, '5_day_training_gaby_downsampled_y_train.parquet.gzip')
y_test_path = os.path.join(
    dataset_dir, '5_day_training_gaby_downsampled_y_test.parquet.gzip')


X_train = pd.read_parquet(X_train_path)
X_test = pd.read_parquet(X_test_path)
y_train = pd.read_parquet(y_train_path)
y_test = pd.read_parquet(y_test_path)

X_train = X_train[::100]
X_test = X_test[::100]
y_train = y_train[::100]
y_test = y_test[::100]

# define space - parameters to tune

space = {
    'sequence_length': hp.choice('sequence_legnth',  range(30, 181, 30)),
    'units': hp.choice('units', range(32, 129, 32)),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd'])
}

trials = Trials()


def objective(params):
    sequence_length = params['sequence_length']
    units = params['units']
    learning_rate = params['learning_rate']
    optimizer = params['optimizer']

    # set up model
    model = StackedLSTM(sequence_length=sequence_length,
                        num_features=X_train.shape[1],
                        lstm_units=units)

    # compile model by optimizer

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=5)

    loss = model.evaluate(X_test, y_test)

    results = {'loss': loss,
               'status': STATUS_OK}
    return results


def run_trials():
    best_trial = fmin(fn=objective,
                      space=space,
                      algo=tpe.suggest,
                      max_evals=10,
                      trials=trials
                      )
    print(best_trial)
    return best_trial


if __name__ == '__main__':
    run_trials()
