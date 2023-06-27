import os
import pandas as pd
import tensorflow as tf
from src.models.StackedLSTM import StackedLSTM
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

X_train = X_train[::1000]
X_test = X_test[::1000]
y_train = y_train[::1000]
y_test = y_test[::1000]


# parameters to tune and define search space
# Define search space
space = {
    'sequence_length': hp.choice('sequence_length', [30, 60, 90, 120, 150, 180]),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd'])
}
# Define trials object
trials = Trials()

# define objective function


def objective(params):
    sequence_length = params['sequence_length']
    learning_rate = params['learning_rate']
    optimizer = params['optimizer']

    # Define model
    model = StackedLSTM(sequence_length,
                        num_features=X_train.shape[1],
                        lstm_units=32)

    # Compile model
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # Evaluate model
    loss = model.evaluate(X_test, y_test)

    return {'loss': loss, 'status': STATUS_OK}


def run_trials():
    # Run hyperparameter search
    best = fmin(objective, space, algo=tpe.suggest,
                max_evals=10, trials=trials)
    print(best)
    return best


if __name__ == '__main__':
    # Run hyperparameter search
    run_trials()
