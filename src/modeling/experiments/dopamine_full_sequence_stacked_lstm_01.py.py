import atexit
import logging
import numpy as np
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from src.data_processing.processors.TrainingProcessor import TrainingProcessor
from src.models.StackedLSTM import StackedLSTM
from src.utilities.os_helpers import set_up_directories
from src.utilities.tensorflow_helpers import set_tensorboard


@atexit.register
def early_exit():
    logging.info("experiment terminated early")


def read_in_process_data(path_to_data, path_to_save_meta_data):
    data = pd.read_parquet(path_to_data)
    logging.info(f'Data loaded from {path_to_data}')
    train_processor = (TrainingProcessor(data)
                       .query_sensor_and_sort_trials_by_subject(sensor='DA')
                       .split_train_val_test_by_subject(target='signal', shuffle=True)
                       .save_subjects_by_category(path_to_save_meta_data)
                       )
    logging.info('Data processed')
    return train_processor


def experiment(processor,
               model_id,
               tensorboard_dir,
               model_save_dir):
    logging.info(f'Running experiment {model_id}')
    tensorboard_callback = set_tensorboard(
        save_directory=tensorboard_dir, model_id=model_id)

    model = StackedLSTM(
        sequence_length=processor.data.time.nunique(),
        num_features=processor.train_x.columns.nunique(),
        lstm_units=processor.train_x.columns.nunique() * 2
    )
    logging.info('Model compiled')
    model.compile(optimizer='adam', loss='mse', metrics=[
        'mae', 'mse', 'mape', 'cosine_similarity'])

    model.fit(processor.train_x,
              processor.train_y,
              epochs=50,
              validation_data=(processor.val_x, processor.val_y),
              callbacks=[tensorboard_callback]
              )
    logging.info('Model fit')

    model.evaluate(processor.test_x, processor.test_y)
    logging.info('Model evaluated')
    model.save(os.path.join(model_save_dir, model_id))
    logging.info('Model saved')
    logging.info(f'Experiment {model_id} complete')


def main():
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline_full_dataset/datasets/full_dataset.parquet.gzip'
    # where all experiment results are saved
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "dopamine_full_sequence_stacked_lstm_01"

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
    dopamine_processor = read_in_process_data(DATA_PATH, EXPERIMENT_DIR)
    experiment(processor=dopamine_processor,
               model_id=EXPERIMENT_NAME,
               tensorboard_dir=TENSORBOARD_DIR,
               model_save_dir=MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
