import atexit
import logging
import numpy as np
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from src.data_processing.processors.LSTMPipe import LSTMPipe
from src.data_processing.processors.TrainingProcessor import TrainingProcessor
from src.models.experimental_dropout_StackedLSTM import StackedLSTM
from src.utilities.os_helpers import set_up_directories
from src.utilities.tensorflow_helpers import set_tensorboard


@atexit.register
def early_exit():
    logging.info("experiment terminated early")


def process_data(path_to_data, path_to_save_meta_data):
    processor_pipe = (LSTMPipe(path_to_data)
    .read_raw_data(sort_by=['mouse_id','sensor','event', 'trial_count'])
    .split_data(processed_data = False, 
                test_size=0.3,
                test_dev_size=0.5, 
                split_group = "mouse_id", 
                stratify_group = "sex", 
                target='signal', 
                save_subject_ids=True, 
                path_to_save =path_to_save_meta_data)
    .transorm_data()
)
    logging.info('Data processed')
    return processor_pipe


def experiment(processor,
               model_id,
               tensorboard_dir,
               model_save_dir):
    logging.info(f'Running experiment {model_id}')
    
    # schedular
    def lr_schedular(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    model = StackedLSTM(
        sequence_length=processor.raw_data['time'].nunique(),
        num_features=processor.X_train.shape[1],
        lstm_units=processor.X_train.shape[1] * 2
    )
    logging.info('Model compiled')
    
    model.compile(optimizer='adam', loss='mse', metrics=[
        'mae', 'mse', 'mape', 'cosine_similarity'])
    
    # call backs
    tensorboard_callback = set_tensorboard(
    save_directory=tensorboard_dir, model_id=model_id)
    
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_schedular, verbose=1)

    model.fit(processor.X_train,
              processor.y_train,
              epochs=200,
              validation_data=(processor.X_dev, processor.y_dev),
              callbacks=[tensorboard_callback, learning_rate_callback]
              )
    logging.info('Model fit')

    model.evaluate(processor.X_test, processor.y_test)
    logging.info('Model evaluated')
    model.save(os.path.join(model_save_dir, model_id))
    logging.info('Model saved')
    logging.info(f'Experiment {model_id} complete')


def main():
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    # where all experiment results are saved
    MAIN_DIR = '/projects/p31961/ENIGMA/results/experiments'
    EXPERIMENT_NAME = "dopamine_full_sequence_stacked_lstm_03"

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
    processor = process_data(DATA_PATH, EXPERIMENT_DIR)
    experiment(processor=processor,
               model_id=EXPERIMENT_NAME,
               tensorboard_dir=TENSORBOARD_DIR,
               model_save_dir=MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
