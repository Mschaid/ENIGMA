from src.processors.Preprocessor import Preprocessor



import pandas as pd
import os
import numpy as np



PATH = '/home/mds8301/gaby_test/processors/unit_test_processor.pkl'
PATH_TO_SAVE = '/projects/p31961/dopamine_modeling/data/prototype_data'
proc = Preprocessor().load_processor(PATH)
proc.one_hot_encode(labels = ['event', 'sensor'])
print('downsampling data')
downsample_data = proc.processed_data.drop(proc.processed_data.index[::200])
print('saving data')
downsample_data.to_parquet(os.path.join(PATH_TO_SAVE, 'downsampled_data.parquet.gzip'), compression='gzip')

query = 'mouse_id == "310_909" & sensor_DA == 1 & event_avoid==1'
non_useful_columns=['mouse_id','event_cue','event_escape', 'event_shock','sensor_D1', 'sensor_D2']
print('filtering data')
mouse_909_DA_avoid = downsample_data.query(query).drop(columns=non_useful_columns)
print('saving data')
mouse_909_DA_avoid.to_parquet(os.path.join(PATH_TO_SAVE, 'mouse_909_DA_avoid.parquet.gzip'), compression='gzip')

# trials_under_5 = mouse_909_DA_avoid.query('trial<=5')
# trials_over_5 = mouse_909_DA_avoid.query('trial>5')

# X_train, y_train = trials_under_5.drop(columns = ['signal']), trials_under_5.signal
# X_test, y_test = trials_over_5.drop(columns = ['signal']), trials_over_5.signal