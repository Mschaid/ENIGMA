import logging
import numpy as np
import pprint
import os
import glob
import pandas as pd
import re

from loguru import logger


class FileScraper:
    def __init__(self, directory: str = None) -> None:

        if directory is not None:
            self._directory = directory
            self.fetch_all_file_names()
            self.file_search_results = self._all_files

        else:
            self._directory = None
            self._all_files = None

        self._extensions_found = []
        self._extensions_not_found = []
        self._keywords_found = []
        self._keywords_not_found = []

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory: str):
        self._directory = directory
        self.fetch_all_file_names()

    @property
    def all_files(self):
        return self._all_files

    @all_files.setter
    def all_files(self, value):
        self._all_files = value

    @property
    def extensions_found(self):
        if self._extensions_found == ['']:
            return []
        else:
            return self._extensions_found

    @extensions_found.setter
    def extensions_found(self, value):
        self._extensions_found = value

    @property
    def extensions_not_found(self):
        return self._extensions_not_found

    @extensions_not_found.setter
    def extensions_not_found(self, value):
        self._extensions_not_found = value

    @property
    def keywords_found(self):
        if self._keywords_found == ['']:
            return []
        return self._keywords_found

    @keywords_found.setter
    def keywords_found(self, value):
        self._keywords_found = value

    @property
    def keywords_not_found(self):
        return self._keywords_not_found

    @keywords_not_found.setter
    def keywords_not_found(self, value):
        self._keywords_not_found = value

    @property
    def file_search_results(self):
        if self.extensions_found is None or self.keywords_found is None:
            return []
        elif len(self.extensions_found) == 0 and len(self.keywords_found) == 0:
            return self.all_files
        else:
            return self._file_search_results

    @file_search_results.setter
    def file_search_results(self, value):
        self._file_search_results = value

    @property
    def search_results(self):
        search_results = {"directory": self.directory,
                          "file_search_results": self.file_search_results,
                          "extensions_found": self.extensions_found,
                          "extensions_not_found": self.extensions_not_found,
                          "keywords_found": self.keywords_found,
                          "keywords_not_found": self.keywords_not_found}
        return search_results

    def fetch_all_file_names(self):
        """
        Function to fetch all file names in a directory and store them in a list, can be used to reset the file names stored in the class.
        """
        all_file_names = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                all_file_names.append(os.path.join(root, file))
        self._all_files = all_file_names
        # self.update_search_results(file_names=self.file_names)

    def filter_files_by_extention(self, *extensions):
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

        formatted_extensions = [
            self.format_input(ext) for ext in extensions]

        # filters files by extension in in the stored file names of the directory
        filtered_files = [
            file for file
            in self.all_files
            if any(file.endswith(ext) for ext in formatted_extensions
                   # if any(ext in file for ext in formatted_extensions
                   )
        ]

        extensions_found = [ext for ext
                            in formatted_extensions
                            if any(ext in file for file in filtered_files)
                            ]

        extensions_not_found = [ext for ext
                                in formatted_extensions
                                if ext not in extensions_found
                                ]

        if len(extensions_found) > 0:
            self.extensions_found = extensions_found

        elif len(extensions) == 0:
            self.extensions_found = []

        else:
            self.extensions_found = None

        self.extensions_not_found = extensions_not_found
        self._file_search_results = filtered_files

        # if no files are found with the specified extensions, all files in directory still saved and info is logged

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

        formatted_keywords = [
            self.format_input(word, format_type="keyword")
            for word in keywords
        ]

        def filter_files_for_keywords(files_to_filter, keywords):
            filtered_files = [file for file in files_to_filter if any(
                word in file for word in keywords)]
            return filtered_files

        if self.extensions_found == []:
            filtered_files = filter_files_for_keywords(
                self.all_files, formatted_keywords)

        elif self.extensions_found is None:
            return []
        else:
            filtered_files = filter_files_for_keywords(
                self.file_search_results, formatted_keywords)

        keywords_found = [word for word
                          in formatted_keywords
                          if any(
                              word in file
                              for file in self.file_search_results
                          )
                          ]

        keywords_not_found = [word for word
                              in formatted_keywords
                              if word not in keywords_found]

        self._file_search_results = filtered_files

        if len(keywords_found) > 0:
            self.keywords_found = keywords_found

        elif len(keywords) == 0:
            self.keywords_found = []
        else:
            self.keywords_found = None

        self.keywords_not_found = keywords_not_found

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
        if input == '':
            return ''

        formatted_input = (
            input
            .strip()
            .strip(",")
        )

        if format_type == "extension":
            formatted_input = formatted_input if formatted_input.startswith(
                ".") else f".{formatted_input}"
            return formatted_input
        elif format_type == "keyword":
            return formatted_input

    def format_user_input(self, user_input: str) -> list:
        """

        Formats the user input by splitting it into a list of strings.
        Parameters:
            user_input (str): The input provided by the user.
        Returns:
            list: A list of strings after splitting the input by spaces or commas.
        """
        # if user_input is None:
        #     return []

        if " " in user_input:
            user_input = user_input.split(" ")
        elif "," in user_input:
            user_input = user_input.split(",")
        else:
            user_input = [user_input]
        return user_input

    def scrape_directory(self, directory: str = '', file_extensions: str = '', keywords: str = '') -> None:

        if directory != '':
            self.directory = directory

        formatted_extensions = self.format_user_input(file_extensions)
        self.filter_files_by_extention(*formatted_extensions)

        formatted_keywords = self.format_user_input(keywords)
        self.filter_files_by_keywords(*formatted_keywords)
