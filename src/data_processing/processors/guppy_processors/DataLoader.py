
import yaml

from src.data_processing.processors.FileScraper import FileScraper
from pathlib import Path


class DataLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.filescraper = FileScraper()
        self._meta_data = None
        self._data_files = None

    @property
    def meta_data(self):
        if self._meta_data is None:
            self._meta_data = self._load_config()
        return self._meta_data

    @property
    def data_files(self):
        return self.filescraper.file_search_results
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def fetch_data_files(self, keywords, file_extensions):
        self.filescraper.scrape_directory(
            directory=self.meta_data['data_path'],
            keywords=keywords,
            file_extensions=file_extensions
        )
