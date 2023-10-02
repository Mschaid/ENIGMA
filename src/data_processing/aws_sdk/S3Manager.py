import boto3
from loguru import logger
from typing import Tuple
from src.data_processing.processors.FileScraper import FileScraper


class S3Manager:
    def __init__(self):

        self._s3_connection = None
        self.scraper = FileScraper()

    def connect_to_s3(self):
        """
        Establishes a connection to the Amazon S3 service.

        Initializes the `s3_connection` attribute of the current instance with a client object
        from the boto3 library, which enables communication with the Amazon S3 service. The client
        object is created using the `boto3.client()` method, passing in the string 's3' as the
        service name.

        Returns
        -------
        None
        """

        try:
            self._s3_connection = boto3.client('s3')
        except Exception as e:
            print(f"Error: {e}")

    @property
    def s3_connection(self):
        """
        Get the S3 connection.

        Returns:
            The S3 connection object.
        """
        if self._s3_connection is None:
            print(f"Connection not established: Calling connect_to_s3() now.")
            self.connect_to_s3()
        return self._s3_connection

    # @property
    # def s3_buckets(self):
    #     """
    #     Get the S3 buckets.

    #     Returns:
    #         The S3 buckets.
    #     """
    #     return self.s3_connection.list_buckets()

    def get_available_buckets(self):
        """
        Retrieves a list of available S3 buckets.

        Prints the name of each bucket.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs while listing the buckets.
        """

        try:
            response = self.s3_connection.list_buckets()
            for bucket in response['Buckets']:
                print(bucket['Name'])
        except Exception as e:
            print(f"Error: {e}")

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

        # self.scraper.directroy = directory
        file_extensions = file_extensions or input(
            "Enter the file extensions to search seperated by a comma or space: ")

        keywords = keywords or input(
            "Enter the keywords to search seperated by a comma or space: ")

        file_extensions = format_user_input(file_extensions)
        keywords = format_user_input(keywords)

        self.scraper.filter_files_by_extention(*file_extensions)
        self.scraper.filter_files_by_keywords(*keywords)
