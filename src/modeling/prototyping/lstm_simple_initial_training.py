import os
import tensorflow as tf
import pandas as pd


from src.modeling.prototyping.lstm_protype_simple import (build_lstm,
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
    
    MODEL_ID = 'lstm_simple_initial_training_with_downsampling'
    MODEL_PATH_TO_SAVE = '/projects/p31961/dopamine_modeling/results/models/'
    
    TENSORBOARD_CALLBACK = set_tensorboard(MODEL_ID)
    
    X_train_path = build_path(PATH_TO_DATA, 'X_train_downsampled')
    y_train_path = build_path(PATH_TO_DATA, 'y_train_downsampled')
    X_test_path = build_path(PATH_TO_DATA, 'X_test_downsampled')
    y_test_path = build_path(PATH_TO_DATA, 'y_test_downsampled')
    
    
    # # read data from parquet files
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path) 
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)
    


    # # #build lodel
    model = build_lstm(sequence_length=90, input_dimentions=X_train.shape[1])
    train_model(model, X_train, y_train, TENSORBOARD_CALLBACK)
    evaluate_model(model, X_test, y_test, TENSORBOARD_CALLBACK)
    inference(model, X_test)
    tf.keras.save_model(model, os.path.join(MODEL_PATH_TO_SAVE, MODEL_ID))
    
