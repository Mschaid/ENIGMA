import logging
from pathlib import Path
from typing import NewType, Protocol

import h5py
import polars as pl
import pretty_errors
import yaml

GuppyOuputPath = NewType("GuppyOuputPath", Path)


# set up logger for data fetcher
logger = logging.getLogger('DataFetcher')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataFetcher(Protocol):
    def __init__(self, path_to_read: Path):
        raise NotImplementedError

    def read_metadata(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError


class GuppyDataFetcher:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._metadata = None

    def _instatntiate_path_objs(self, metadata):
        for key in metadata.keys():
            if "path" in key:
                metadata[key] = [Path(p) for p in metadata[key]]
        return metadata

    def read_metadata(self):

        try:
            with open(self.output_path/"metadata.yaml") as f:
                raw_metadata = yaml.load(f, Loader=yaml.FullLoader)
                metadata = self._instatntiate_path_objs(raw_metadata)

        except FileNotFoundError as e:
            logger.error(
                f"metadata.yaml not found in {self.output_path.name}, returned None")
            metadata = None

        return metadata

    @ property
    def metadata(self):
        if self._metadata is None:
            metadata = self.read_metadata()
            self._metadata = metadata
        return self._metadata

    #! build in data reader for metadata keys that have path in them
    def filter_metadata_for_keywords(self, *keywords):
        filtered_data = {
            k: v for k, v in self.metadata.items()
            if any(kw in k for kw in keywords)
        }
        return filtered_data
