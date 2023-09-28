import logging
import numpy as np
import pprint
import os
import glob
import pandas as pd
import re

from loguru import logger


# TODO implement loguru


class FileScraper:
    def __init__(self, data_directory: str = None):

        if data_directory is not None:
            self._directory = data_directory
        else:
            self._directory = None
        self._file_names = None

    @property
    def directory(self):
        if self._directory is None:
            self._directory = input("Enter the directory of the data files: ")
        return self._directory

    @directory.setter
    def directory(self, value):
        self._directory = value

    def set_directory(self, directory=None):
        if directory is None:
            self.directory
        self._directory = directory

    @property
    def file_names(self):
        if self._file_names is None:
            self.fetch_all_file_names()
        return self._file_names

    @file_names.setter
    def file_names(self, value):
        self._file_names = value

    def fetch_all_file_names(self):
        all_file_names = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                all_file_names.append(os.path.join(root, file))
        self.file_names = all_file_names

    def filter_files_by_extention(self, *extensions):
        filtered_files = filter(
            lambda file: file.endswith(extensions), self.file_names)
        filtered_files = list(filtered_files)

        try:
            assert len(filtered_files) > 0
            self._file_names = filtered_files
            logger.info(f"files found: {self._file_names}")

        except AssertionError as e:
            logger.info(f"No files found with the specified extention(s).")

    def filter_files_by_keywords(self, *keywords):
        filtered_files = filter(lambda list_: any(
            keyword in list_ for keyword in keywords), self.file_names)
        filtered_files = list(filtered_files)
        try:
            assert len(filtered_files) > 0
            self._file_names = filtered_files
            logger.info(f"files found: {self._file_names}")

        except AssertionError as e:
            logger.info(f"No files found with the specified extention(s).")
