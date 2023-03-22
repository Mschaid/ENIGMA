import logging
import numpy as np
import os
import glob
import pandas as pd
import re
import sys
import time
import typing
import pickle

from src.utilities.exceptions import *
from src.utilities.os_helpers import create_new_directoy


class SchemaBuilder:
    def __init__(self, data_directory: str = None):
        self.self = self
        self.data_directory = data_directory

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

    def collect_data_files(self, search_keys: list, file_types: list):
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
                    filenames.append(os.path.join(root, file))
            return filenames

        # helper function to filter a list by a list of terms
        def filter_list_by_terms(list_to_search, terms):
            filtered_list = filter(lambda list_: any(
                re.search(term, list_) for term in terms), list_to_search)
            return filtered_list

        # collect all file names in the data directory
        self.data_files = collect_filenames(self.data_directory)
        self.matched_file_types = list(filter_list_by_terms(
            self.data_files, file_types))  # filters all of the files for file types
        # filters all of the matched file types for search keys
        self.matched_search_keys = list(
            filter_list_by_terms(self.matched_file_types, search_keys))

    def aggregate_data(self, files: list = None):
        """ # Summary:
        Method for aggregating the data files into a single dataframe. Current filetypes supported are csv, xlsx, feather, pkl, and h5.
        csv, xlsc, feather, pickle and h5

        ## Args: 
            files (list): a list of files to be aggregated, if None, all files in the matched search keys attribute are aggregated 

        ##Raises:
            FileTypeError: if file type is not supported, FileTypeError is raised
        """

        if files is None:
            files = self.matched_search_keys

        self.data_frames = []
        for file in files:
            file_path = os.path.join(self.data_directory, file)
            # try:
            if file_path.endswith('csv'):
                df = pd.read_csv(file_path)
                self.data_frames.append(df)

            elif file_path.endswith('xlsx'):
                df = pd.read_excel(file_path)
                self.data_frames.append(df)

            elif file_path.endswith('feather'):
                df = pd.read_feather(file_path)
                self.data_frames.append(df)

            elif file_path.endswith('pkl'):
                df = pd.read_pickle(file_path)
                self.data_frames.append(df)

            elif file_path.endswith('h5'):
                df = pd.read_hdf(file_path)
                self.data_frames.append(df)

            else:
                raise FileTypeError(filetype=file_path.split('.')[-1])

        self.aggregate_dataframes = pd.concat(self.data_frames)

    def save_schema(self, file_path: str = None, new_dir_extension: str = None, file_name: str = None):
        """            
         # Summary
        saves the schema object to a pick file, can be loaded later. 

        Args:
            file_path (str):path for the file to be saved
            file_str (str): name of the file to be saved

        """
        if file_path is None:
            file_path = self.data_directory

        if new_dir_extension is None:
            new_dir_extension = 'saved_objects'

        if file_name is None:
            file_name = 'data_schema.pkl'
        else:
            file_name = file_name + '.pkl'

        # new directory to be created
        new_directory = os.path.join(file_path, new_dir_extension)
        # file path for the saved object
        self.stored_object_file_path_ = os.path.join(new_directory, file_name)

        # create the new directory if it does not exist
        if os.path.exists(new_directory):
            pass
        else:
            os.mkdir(new_directory)

        with open(self.stored_object_file_path_, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_schema(cls, file_path):
        """
        # Summary
        loads the schema object from a pick file, can be loaded later. 

        Args:
            file_path (str):path for the file to be saved

        """
        with open(file_path, 'rb') as f:
            schema = pickle.load(f)
        return schema


# TODO delete this later
if __name__ == '__main__':
    DIR = r"C:\Users\mds8301\Documents\Github\dopamine_modeling\data"
    sb = SchemaBuilder.load_schema(
        file_path=r'C:\Users\mds8301\Documents\Github\dopamine_modeling\data\test_save_schema\data_schema.pkl')
    print(sb.data_directory)
