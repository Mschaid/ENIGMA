import os
import tensorflow as tf
import pandas as pd


from src.model_experimentation.prototypes.lstm_protype_simple import (build_lstm,
                                                                      set_tensorboard,
                                                                      train_model,
                                                                      evaluate_model,
                                                                      validated_tf,
                                                                      inference
                                                                      ) 




def build_path(path_to_data,dataset_name, data_set_prefix = '5_day_training_gaby_', suffix = '.parquet.gzip'):
    path =  os.path.join(path_to_data, f'{data_set_prefix}{dataset_name}{suffix}')
    return path

def read_data_from_paths(*paths):
    for path in paths:
        yield pd.read_parquet(path)



if __name__ == '__main__': 
 

    # print tensorflow specs
    validated_tf()
    # set global variables
    PATH_TO_DATA = '/projects/p31961/gaby_data/aggregated_data/data_pipeline/datasets'
    
    MODEL_ID = 'lstm_simple_initial_training'
    MODEL_PATH_TO_SAVE = '/projects/p31961/dopamine_modeling/results/models/'
    
    TENSORBOARD_CALLBACK = set_tensorboard(MODEL_ID)
    
    X_train_path = build_path(PATH_TO_DATA, 'X_train')
    y_train_path = build_path(PATH_TO_DATA, 'y_train')
    X_test_path = build_path(PATH_TO_DATA, 'X_test')
    y_test_path = build_path(PATH_TO_DATA, 'y_test')
    
    # # read data from parquet files
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path)
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)
    
    training_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
    testing_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))
    
    training_output_path = os.path.join(PATH_TO_DATA, 'tensorflow_training_dataset')
    testing_output_path = os.path.join(PATH_TO_DATA, 'tensorflow_testing_dataset')
    # save dataset to tfrecord
    training_dataset.save(training_output_path)
    testing_dataset.save(testing_output_path)
