
import logging
import numpy as np
import os
import glob
import pandas as pd
import re
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
        
        
    def collect_data_files(self,search_keys:list, file_types:list):
        """
        # Summary:
        Method for collecting the data files from the data directory

        ## Args:
            search_keys (list): a list of search keys to filter the data files
            file_types (list): a list of file types to filter the data files

        ## Attributes:
            data_files (list): a list of all files in the data directory
            matched_file_types (filter_object): a list of files that match the file types
            matched_search_keys (filter_object): a list of files that match the search keys (and file types)
        Returns:
            None
        """


        # first search for search_keys in the file directory        
        
        def collect_filenames(directory):
            filenames = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filenames.append(file)
            return filenames

        def filter_list_by_terms(list, terms):
            filtered_list = filter(lambda l: any(re.search(t,l) for t in terms), list)
            return filtered_list
        
        self.data_files = collect_filenames(self.data_directory)
        self.matched_file_types = filter_list_by_terms(self.data_files,file_types)
        self.matched_search_keys = filter_list_by_terms(self.matched_file_types,search_keys)
        
        return 
        

        
        # then search for file_types in the file directory
    
        
        # csv, feather, pickle, parquet
        # for files in data_files:
        #     if file 
        
        
    def aggregate_data(self, filetype:str, ):
        pass
    
    
    def save_schema(self):
        pass
    
    def load_schema(self):
        pass
    
    