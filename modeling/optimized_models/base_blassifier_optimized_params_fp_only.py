import json
import os
import pandas as pd
import tensorflow as tf
import logging


from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from src.utilities.os_helpers import set_up_directories
from src.utilities.tensorflow_helpers import set_tensorboard

from src.models.BaseClassifier import BaseClassifier

""" 
This model was optimzied using hyperopt in base_classifier_tuning.py

"""


def process_data(file_path, subject_ids_path, features_to_drop):
    """
    Process the data from a given file path and save the processed data to a specified path.

    Args:
        file_path (str): The path to the file containing the data.
        path_to_save (str): The path to save the processed data.

    Returns:
        ClassifierPipe: The processed data.
    """

    processor = (ClassifierPipe(file_path)
                 .read_raw_data()
                 .calculate_max_min_signal()
                 .stratify_and_split_by_mouse(load_subject_ids=True,
                                              subject_ids_path=subject_ids_path,
                                              target='action')
                 .drop_features(features_to_drop)
                 .transform_data(numeric_target_dict={'avoid': 1, 'escape': 0})
                 )
    return processor


def load_parameters(best_params_file_path):
    """
    Load the parameters from the specified file path.

    Parameters
    ----------
    best_params_file_path : str
        The file path to the JSON file containing the parameters.

    Returns
    -------
    params : dict
        The parameters loaded from the file.
    """
    results = json.load(open(best_params_file_path))
    return results["params"]


def train_optimized_model(processor, best_params, tensorboard_dir, model_name):
    """
    Train an optimized model using the given processor, best parameters, directory to save,
    and model name.

    Parameters
    ----------
    processor : Processor
        The processor object containing the training data.
    best_params : dict
        A dictionary containing the best hyperparameters for the model.
    directory_to_save : str
        The directory where the trained model will be saved.
    model_name : str
        The name of the model.

    Returns
    -------
    model : BaseClassifier
        The trained model.
    """

    tensorboard_callback = set_tensorboard(
        save_directory=tensorboard_dir, model_id=model_name)

    model = (BaseClassifier(
        number_of_layers=best_params["number of layers"],
        number_of_units=best_params["number of units"],
        dropout_rate=best_params["dropout rate"])
    )
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc-roc')]

    model.compile(optimizer=best_params["optimizers"],
                  loss='binary_crossentropy',
                  metrics=metrics)

    model.fit(processor.X_train,
              processor.y_train,
              batch_size=best_params["batch size"],
              epochs=best_params["epochs"],
              validation_data=(processor.X_dev, processor.y_dev),
              callbacks=[tensorboard_callback])

    return model


def main():
    """
    Runs the main function of the program.

    This function sets up the necessary directories for logging, training results, and model saving.
    It then processes the data, loads the optimized parameters, trains the model, and finally saves the model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # data path and main directioes to save logs, training results and model
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'

    FEATURES_TO_DROP = ["mouse_id", "event",
                        "sex", "day", "trial_count", "trial"]
    SUBJECT_IDS_PATH = "/projects/p31961/ENIGMA/results/optimized_models/subjects.json"

    PARAM_FILE_PATH = '/projects/p31961/ENIGMA/results/experiments/base_classifier_tuning/best_params.json'
    EXPERIMENT_MAIN_DIR = '/projects/p31961/ENIGMA/results/optimized_models'
    MODEL_SAVE_MAIN_DIR = '/projects/p31961/ENIGMA/results/models/optimzied_models'
    MODEL_NAME = "BaseClassifier_optimized_fp_only"

    # directories to create
    EXPERIMENT_DIR = os.path.join(EXPERIMENT_MAIN_DIR, MODEL_NAME)
    TENSOR_BOARD_DIR = os.path.join(EXPERIMENT_DIR, 'tensorboard')
    MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_MAIN_DIR, MODEL_NAME)
    set_up_directories(EXPERIMENT_DIR, MODEL_SAVE_DIR, TENSOR_BOARD_DIR)

    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{MODEL_NAME}.log')
    logging.basicConfig(filename=LOG_FILE_PATH,
                        filemode='w',
                        level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info(
        f'Created new directories: {EXPERIMENT_MAIN_DIR}, {MODEL_SAVE_MAIN_DIR}')

    logging.info('Processing data')
    processor = process_data(DATA_PATH, SUBJECT_IDS_PATH, FEATURES_TO_DROP)
    logging.info('Loading optimzied paramteres')
    best_params = load_parameters(PARAM_FILE_PATH)
    logging.info('Training model')
    model = train_optimized_model(processor=processor, best_params=best_params,
                                  tensorboard_dir=TENSOR_BOARD_DIR, model_name=MODEL_NAME)
    logging.info('Saving model')
    model.save(EXPERIMENT_DIR)
    logging.info(f'{MODEL_NAME} training and saving complete')


if __name__ == "__main__":
    main()
