
from src.processors.Preprocessor import Preprocessor

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split
import logging

# import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.losses import BinaryCrossentropy




def experiment():
    # FEATURES = ['day', 'time', 'trial',
    #             'signal', 'event_cue', 'event_shock',
    #             'sensor_D1', 'sensor_D2', 'sensor_DA']
    # TARGET = 'signal'

    PATH_TO_PROCESSED_DATA = input('Enter path to processed data: ')
    print(PATH_TO_PROCESSED_DATA)
    
    # processor = Preprocessor(PATH_TO_PROCESSED_DATA)
    # print(processor.y_train)
    
    

if __name__ == "__main__":
    experiment()
