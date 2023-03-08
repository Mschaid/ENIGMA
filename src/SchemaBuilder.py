
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import typing



class SchemaBuilder:
    def __init__(self, logger):
        self.self = self
        self.logger = logger

    def request_directory(self):
        """
        # Summary:
        Method for requesting the directory of the data files from the user.
        
        ## Agrs:
        None: user is promoted to enter the directory of the data files
        
        ## Attributes:
        data_directory (str): the directory of the data files
        
        ### Logs:
        info: the directory of the data files
        
        ## Returns:
        None
        """
        
        
        self.data_directory = input("Enter the directory of the data files: ")
        self.logger.info(f"Data directory: {self.data_directory}")
        
        
    def collect_data_files(self,search_keys:list, *file_regex:str):
        
        for key in search_keys:
        
        # csv, feather, pickle, parquet
        
        
        
    def aggregate_data(self, filetype:str, ):
        pass
    
    
        