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
    def __init__(self, data_directory: str = None) -> None:

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

    def set_directory(self, directory: str = None):
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
        """
        Function to fetch all file names in a directory and store them in a list, can be used to reset the file names stored in the class.
        """
        all_file_names = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                all_file_names.append(os.path.join(root, file))
        self.file_names = all_file_names

    def filter_files_by_extention(self, *extensions: str):
        """
        Filters the files in the directory by the specified extensions.

        Parameters
        ----------
        *extensions : str
            The extensions to filter the files by.

        Returns
        -------
        None
        """

        # formats extensions if not already formatted correctly
        formatted_extensions = [
            self.format_input(ext) for ext in extensions]

        # filters files by extension in in the stored file names of the directory
        filtered_files = [file for file in self.file_names if any(
            ext in file for ext in formatted_extensions)]

        # checks if any files were found with the specified extensions, if not, looged and keeps all files in directory
        try:
            assert len(filtered_files) > 0
            extensions_found = [ext for ext in formatted_extensions if any(
                ext in file for file in filtered_files)]
            extensions_not_found = [
                ext for ext in formatted_extensions if ext not in extensions_found]

            # loggs the files found
            logger.info(
                f"Files filtered by extensions found: {filtered_files}")
            # what ever extensions are not found are logged
            if len(extensions_not_found) > 0:
                logger.info(
                    f"Files of extension type: {extensions_not_found} not found.")
            else:
                pass

        # updates _file_names stored as the filtered_files
            self._file_names = filtered_files
        # if no files are found with the specified extensions, all files in directory still saved and info is logged
        except AssertionError as e:
            logger.info(
                f"No files found with the specified extention(s), all files in directory returned.")

    def filter_files_by_keywords(self, *keywords: str):
        """
        Filters the files in the directory by the specified keywords.

        Parameters
        ----------
        *keywords : str
            The keywords to filter the files by.

        Returns
        -------
        None
        """

        # formats keywords into list for searching
        formatted_keywords = [self.format_input(
            word, format_type="keyword") for word in keywords]
        print(formatted_keywords)

        filtered_files = [file for file in self.file_names if any(
            word in file for word in formatted_keywords)]

        try:
            assert (len(filtered_files) > 0)

            keywords_found = [word for word in formatted_keywords if any(
                word in file for file in filtered_files)]

            keywords_not_found = [
                word for word in formatted_keywords if word not in keywords_found]

            logger.info(f"Files filtered by keywords found: {filtered_files}")
            if len(keywords_not_found) > 0:
                logger.info(
                    f"Files of keyword type: {keywords_not_found} not found.")

            self._file_names = filtered_files

        except AssertionError as e:
            logger.info(
                f"No files found with the specified keyword(s), all files that were searched returned.")

    def format_input(self, input, format_type="extension"):
        """
        Formats the input to be used in the filter_files_by_extention() and filter_files_by_keywords() methods.
        input: str
        format_type: str, default="extension", options=["extension", "keyword"]

        Returns
        -------
        str
            The formatted input.
        """
        formatted_input = (
            input
            .strip()
            .strip(",")
        )

        if format_type == "extension":
            return formatted_input if formatted_input.startswith(".") else f".{formatted_input}"
        elif format_type == "keyword":
            return formatted_input
    def scrape_directoy(self, directory: str = None, file_extensions: str = None, keywords: str = None) -> None:
        """
        Scrapes a directory for files based on specified file extensions and keywords.

        Parameters:
            directory (str): The directory to scrape. If None, the user will be prompted to enter a directory.
            file_extensions (str): The file extensions to search for, separated by commas or spaces. If None, the user will be prompted to enter file extensions.
            keywords (str): The keywords to search for, separated by commas or spaces. If None, the user will be prompted to enter keywords.

        Returns:
            None
        """

        def format_user_input(user_input: str) -> list:
            """

            Formats the user input by splitting it into a list of strings.
            Parameters:
                user_input (str): The input provided by the user.
            Returns:
                list: A list of strings after splitting the input by spaces or commas.
            """
            if " " in user_input:
                user_input = user_input.split(" ")
            elif "," in user_input:
                user_input = user_input.split(",")
            else:
                user_input = [user_input]
            return user_input

        self.directory = directory
        # self.scraper.directroy = directory
        if file_extensions is None:
            file_extensions = None
        else:
            file_extensions = format_user_input(file_extensions)
            self.filter_files_by_extention(*file_extensions)
            

        if keywords is None:
            keywords = None
        else:
            keywords = format_user_input(keywords)
            self.filter_files_by_keywords(*keywords)