import boto3
from loguru import logger
from typing import Tuple

import yaml

from src.data_processing.processors.FileScraper import FileScraper
from pathlib import Path



class DataLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        # self.filescraper = FileScraper()
        # self._meta_data = None

    # @property
    # def meta_data(self):
    #     if self._meta_data is None:
    #         self._meta_data = self._load_config()
    #     return self._meta_data
    
    # def _load_config(self):
    #     with open(self.config_path, 'r') as f:
    #         config = yaml.safe_load(f)
    #     return config


class DataProcessor:
    def __init__(self, meta_data: DataLoader):
        self.meta_data = meta_data


class Visualizer:
    pass


def main():
    CONFIG_PATH = '/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Shudi/LHA_dopamine/conf/config.yaml'
    meta_data = DataLoader(CONFIG_PATH)
    # data_processor = DataProcessor(meta_data)
    # print(meta_data.meta_data)


if __name__ == '__main__':
    main()


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

    