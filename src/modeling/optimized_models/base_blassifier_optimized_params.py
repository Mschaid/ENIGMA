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

def process_data(file_path, path _to_save):
    
    processor = (ClassifierPipe(file_path)
                  .read_raw_data()
                  .calculate_max_min_signal()
                  .split_data(test_size=0.3,
                              test_dev_size=0.5,
                              split_group="mouse_id",
                              stratify_group="sex",
                              target='action',
                              save_subject_ids=True,
                              path_to_save=os.path.dirname(_to_save))
                  .transorm_data(numeric_target_dict={'avoid': 1, 'escape': 0})
                  )
    return processor

def load_parameters(best_params_file_path):
    results = json.load(open(best_params_file_path))
    return results["params"]


def train_optimized_model(processor, best_params): 
    
    tensorboard_callback = set_tensorboard(save_directory=EXPERIMENT_DIR, model_id=EXPERIMENT_NAME)
    
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
              validation_data=(processor.X_dev, processor.y_dev))
    
    
    return model
    
    


        
        
    )

def main():
    
    # data path and main directioes to save logs, training results and model
    DATA_PATH = '/projects/p31961/gaby_data/aggregated_data/raw_data/datasets/raw_data_raw_data.parquet.gzip'
    

    PARAM_FILE_PATH
    EXPERIMENT_MAIN_DIR = '/projects/p31961/ENIGMA/results/optimized_models'
    MODEL_SAVE_MAIN_DIR= '/Users/mds8301/Development/ENIGMA/results/models/optimized_models'
    MODEL_NAME = "BaseClassifier_optimized"
    
    #directories to create
    EXPERIMENT_DIR = os.path.join(EXPERIMENT_MAIN_DIR, MODEL_NAME)
    TENSOR_BOARD_DIR = os.path.join(EXPERIMENT_DIR, 'tensorboard')
    MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_MAIN_DIR, MODEL_NAME)
    set_up_directories(EXPERIMENT_DIR, MODEL_SAVE_DIR), TENSOR_BOARD_DIR
    
    LOG_FILE_PATH = os.path.join(EXPERIMENT_DIR, f'{MODEL_NAME}.log')
    logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w',
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
    
    
    processor = process_data(DATA_PATH, EXPERIMENT_DIR)
    best_params = load_parameters(best_params_file_path)
    model = train_optimized_model(processor, best_params)
    model.save_model(EXPERIMENT_DIR)

    

if __name__ == "__main__":
    main()