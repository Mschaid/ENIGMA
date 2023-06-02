import multiprocessing
import time
import pandas as pd 
import os
import seaborn as sns
import numpy as np
import h5py

from src.model_builders import ModelBuilderBase

PATH = '/projects/p31961/gaby_data/aggregated_data/data_pipeline/datasets/5_day_training_gaby_dataset.h5'

# with h5py.File(PATH, 'r') as f:
#     print(list(f.keys()))
    
class TestBuilder:

    def __init__(self, 
                 model,
                 path_to_processed_data,
                 ):


        self.model = None
        self.path_to_processed_data = path_to_processed_data
        

    def load_preprocessed_data(self, num_processes):

        with h5py.File(self.path_to_processed_data, 'r') as file:
            dataframe_group = file['dataframes']
            keys = list(dataframe_group.keys())
            
            with multiprocessing.Pool(processes = num_processes) as pool:
                results = pool.map(self.load_dataframe, [(self.path_to_processed_data, key) for key in keys])
                
            pool.close()
            pool.join()
                
            for key, dataframe in zip(keys, results):
                setattr(self, key, dataframe)
                
            attrs_group = file['attributes']
            for key in attrs_group.keys():
                setattr(self, key, attrs_group[key][()])
                    
    @staticmethod
    def load_dataframe(args):
        file_path, key = args
        with h5py.File(file_path, 'r') as file:
            dataframe_group = file['dataframes']
            dataset = dataframe_group[key]
            return pd.DataFrame(dataset[:])

                
test = TestBuilder(model=None, path_to_processed_data=PATH)

num_cores = multiprocessing.cpu_count()
num_processes = num_cores - 3
start_time = time.time()
test.load_preprocessed_data(num_processes=num_processes)
end_time = time.time()

print(num_cores)
print(num_processes)
print(f"Time to load data: {end_time - start_time}")
print(dir(test))
print(test.X_train.columns)