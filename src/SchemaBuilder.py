
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import typing



class SchemaBuilder:
    def __init__(self):
        
        
    
    def set_logger_config(self, directory:str, file_name:str, format:str = '[%(asctime)s] %(levelname)s-%(message)s', level= logging.INFO)-> None:
        """# Summary:
        Method for setting the logger configuration

        ### Args:
            directory (str): when method is called, user will be prompted to enter the directory where the log file will be saved
            
            file_name (str): when method is called, user will be prompted to enter the name of the log file
            
            format (str, optional): specific to the logging format,  Defaults to '[%(asctime)s] %(levelname)s-%(message)s'
            see https://docs.python.org/3/library/logging.html?highlight=basicconfig#logging.basicConfig for documentation.
            
            level (_type_, optional): _description_. Defaults to logging.INFO.
            
        ### Attributes:
            logging_directory (str): the directory of the log file
            logger (logging.Logger): the logger object            
        ### Returns:
            None        
        """
        directory = input("Enter the directory of the log file directory: ")
        file_name = input("Enter the file name of the log file: ")
        #TODO decorator for back calculating the directory and file name if logging_directory is changed
        
        self.logging_directory = os.path.join(directory, file_name)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename = self.logging_directory,
            level = level, 
            format = format
            )
        

    def request_directory(self):
        self.data_directory = input("Enter the directory of the data files: ")
        logger.info(f"Data directory: {self.data_directory}")
        
    def aggregate_data(self):
        