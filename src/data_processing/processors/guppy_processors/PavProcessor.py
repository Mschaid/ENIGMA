import yaml
from src.data_processing.processors.FileScraper import FileScraper

from src.data_processing.processors.pavlovian_processors.DataLoader import DataLoader


class PavProcessor:
    def __init__(self, meta_data: DataLoader):
        self.meta_data = meta_data
