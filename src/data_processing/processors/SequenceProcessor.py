import numpy as np
import pandas as pd
import tensorflow as tf


class SequenceProcessor:
    """ Class for processing sequence data for LSTM or similiar models.
    
    Attributes:
    data: pd.DataFrame that has been preprocessed
    
    
    """
    def __init__(self, data):
        self.data = data
        self.subject_cols = None
        
    def batch_by_subject(self, subject_prefix):
    # get all cols with mouse_id
    self.subject_cols = [col for col in self.data.columns if subject_prefix in col]
    
    def query_subject(df, subject):
        return df.query(f'{subject}==1').drop(columns = mouse_ids).reset_index(drop = True)
    
    mouse_ids = [col for col in df.columns if "mouse_id_" in col]
    # create list of dataframes for each mouse
    
    batches = [query_mouse(df, mouse) for mouse in mouse_ids]
    # return list of mouse_ids and list of dataframes
    return mouse_ids, batches
mouse_id, batches = batch_by_mouse(da_data)

