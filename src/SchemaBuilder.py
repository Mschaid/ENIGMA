
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
    def __init__(self, directory: str = None):
        self.self = self
        self.data_directory = directory

    def request_directory(self):
        """
        # Summary:
        Method for requesting the directory of the data files from the user.
        
        ## Agrs:
        None: user is promoted to enter the directory of the data files
        
        ## Attributes:
        data_directory (str): the directory of the data files

        ## Returns:
        None
        """
        
        self.data_directory = input("Enter the directory of the data files: ")

        
        
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


        
        # helper function to collect all file paths within self.data_directory
        def collect_filenames(directory):
            filenames = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filenames.append(file)
            return filenames

        # helper function to filter a list by a list of terms
        def filter_list_by_terms(list_to_search, terms):
            filtered_list = filter(lambda list_: any(re.search(term,list_) for term in terms), list_to_search)
            return filtered_list
        
        self.data_files = collect_filenames(self.data_directory) #collect all file names in the data directory
        self.matched_file_types = list(filter_list_by_terms(self.data_files,file_types)) # filters all of the files for file types
        self.matched_search_keys = list(filter_list_by_terms(self.matched_file_types,search_keys)) # filters all of the matched file types for search keys
     

    def aggregate_data(self): 
        #todo exception handing for file types that don't exist
        #
        # FileTypeError(Exception):
            
        
        for file in self.matched_search_keys:
            try
            if file.endswith('csv'):
                df = pd.read_csv(file)
                
            if file.endswith('xlsx'):
                df = pd.read_excel(file)
                
            if file.endswith('feather'):
                df = pd.read_feather(file)
            
            if file.endswith('pkl'):
                df = pd.read_pickle(file)
            
            if file.endswith('h5'):
                df = pd.read_hdf(file)
            #todo concat dataframes
            #self.dataframe = concat dateframes
      
        
    
    
    def save_schema(self):
        pass
    
    def load_schema(self):
        pass
    
    # how to write tests for SchemaBuilder?
    