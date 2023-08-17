
import os
import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utilities.pandas_helpers import filter_columns_by_search, flatten_dataframe

class ClassifierPipe:
    def __init__(path_to_data):
        self.path_to_data = path_to_data
    
    
    # read raw data
    def read_raw_data(self):
        
        self.raw_data = pd.read_parquet(self.path_to_data)
        return self
    # reduce signal to max and min
    def calculate_max_min_signal(self):
        
        events = filter_columns_by_search(self.raw_data, 'event')
        actions = filter_columns_by_search(self.raw_data, 'action')
        mouse = filter_columns_by_search(self.raw_data, 'mouse')
        sensors = filter_columns_by_search(self.raw_data, 'sensor')
        sex = filter_columns_by_search(self.raw_data, 'sex')
        self.processed_data =  (
            self.raw_data
            .groupby(by = mouse+events+actions+sensors+sex+['day', 'trial_count'], as_index = False).agg({"signal": ["max", "min"]})
            .pipe(flatten_dataframe)
            .rename(columns = lambda c: c.strip("_"))
            .drop(columns ='index')
        )

        return self
        
    # split data into train and test and save subject ids to json
    
    
    # create pipeline
    
    