

import atexit
import logging
import numpy as np
import os
import pandas as pd
import numpy as np
import tensorflow as tf


from src.data_processing.processors.SequenceProcessor import SequenceProcessor
from src.models.BatchStackedLSTM import BatchStackedLSTM
from src.utilities.tensorflow_helpers import set_tensorboard


@atexit.register
def early_termination():  # helper function to log if experiment is terminated early/ errors out
    logging.info('Experiment terminated early')


def set_up_directories(*dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def processes_data(path):
    data = pd.read_parquet(path)  # read data into dataframe
    logging.info(f'Data loaded and processing initiated')
    sequence_processor = (
        SequenceProcessor(data=data)  # load data into sp
        # filter for DA sensor and sort by time
        .query_sensor_and_sort_time_subject(sensor='DA')
        .encode_cyclic_time()  # encode time as sin/cos
        # batch by subject and store
        .batch_by_subject(subject_prefix='mouse_id_')
        # pad all batches with number that doesn't exist in data-> model will learn to ignore
        .pad_batches(value=-1000)
        # split batches and store into attributes
        .split_training_val_test_batches(target='signal')
        .reshape_batches()
    )
    logging.info(f'Data processing complete')
    return sequence_processor


def experiment(processor,
               model_id,
               tensorboard_dir,
               model_save_dir):
    logging.info(f'Running experiment {model_id}')
    tensorboard_callback = set_tensorboard(
        save_directory=tensorboard_dir, model_id=model_id)

    model = BatchStackedLSTM(input_dimensions=processor.train_batches_X.shape,
                             lstm_units=128)
    logging.info(f'Compiling model {model_id}')
    model.compile(optimizer='adam', loss='mse', metrics=[
                  'mae', 'mse', 'mape', 'cosine_similarity'])
    logging.info(f'Fitting model {model_id}')
    model.fit(processor.train_batches_X,
              processor.train_batches_y,
              epochs=50,
              batch_size=processor.train_batches_X.shape[0],
              validation_data=(processor.val_batches_X,
                               processor.val_batches_y),
              callbacks=[tensorboard_callback]
              )
    logging.info(f'Evaluating model {model_id}')
    model.evaluate(processor.test_batches_X, processor.test_batches_y)
    logging.info(f'Saving model {model_id}')
    model.save(os.path.join(model_save_dir, model_id))
    logging.info(f'Experiment {model_id} complete')


def main():

    # set up directories and paths
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'

    # where all experiment results are saved
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = 'dopamine_batched_stacked_lstm'  # name of experiment
    # path to experiment directory
    EXPERIMENT_DIR = os.path.join(MAIN_DIR, EXPERIMENT_NAME)

    # path to save keras moodel
    MODEL_SAVE_DIR = os.path.join(EXPERIMENT_DIR, 'models')
    # path to save tensorboard logs
    TENSORBOARD_DIR = os.path.join(EXPERIMENT_DIR, 'tensorboard')
    LOG_FILE_PATH = os.path.join(
        EXPERIMENT_DIR, f'{EXPERIMENT_NAME}.log')  # path to save log file

    # create new directoriues if they don't exist
    set_up_directories(EXPERIMENT_DIR, MODEL_SAVE_DIR, TENSORBOARD_DIR)

    # set up logger
    logging.basicConfig(filename=LOG_FILE_PATH,
                        filemode='w',
                        level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s - %(message)s')




    logging.info(
        f'Created new directories: {EXPERIMENT_DIR}, {MODEL_SAVE_DIR}, {TENSORBOARD_DIR}')

    # preprocess data

    dopamine_processor = processes_data(DATA_PATH)
    # run experiment
    experiment(processor=dopamine_processor,
               model_id='dopamine_batched_stacked_lstm',
               tensorboard_dir=TENSORBOARD_DIR,
               model_save_dir=MODEL_SAVE_DIR
               )


if __name__ == "__main__":
    main()
