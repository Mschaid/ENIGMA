
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any

"""
This class is used to load data from the guppy experiment.
output folder 
get storesList.csv metadata

get subject id and data from storeslisting.txt file

query files needed from extensions needed from yaml file
timestamp align events
"""


class ExperimentMetaData:

    def __init__(self, main_path: str):
        self.main_path = Path(main_path)  # this is the path from tdt
        self._stores_list: Dict[int, pd.DataFrame] = None
        self._meta_data: Dict[str, Any] = None
        self._guppy_paths: Dict[str, Path] = None

    def _get_output_paths(self) -> List[Path]:
        sub_dirs = [path for path in self.main_path.iterdir(
        ) if path.is_dir() and 'output' in path.name]
        return list(sub_dirs)

    def _get_meta_data(self) -> Dict[str, Any]:
        meta_data = {}
        for file in self.main_path.iterdir():
            if file.is_file() and 'StoresListing.txt' in file.name:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    meta_data_lines = lines[1:5]
                    for line in meta_data_lines:
                        # this space here is needed to precent stripping the time stamp
                        key, value = line.strip('\n').split(': ')
                        meta_data[key] = value
        return meta_data

    @property
    def meta_data(self) -> Dict[str, Any]:
        if self._meta_data is None:
            self._meta_data = self._get_meta_data()
        return self._meta_data

    def _get_stores_list_paths(self) -> List[Path]:
        output_paths = self._get_output_paths()
        stores_lists_paths = [output_path /
                              'storesList.csv' for output_path in output_paths]
        return stores_lists_paths

    @property
    def stores_lists_as_frame(self) -> Dict[str, pd.DataFrame]:
        return {f'output_{i+1}': pd.read_csv(path) for i, path in enumerate(self.guppy_paths['stores_lists_paths'])}

    @property
    def guppy_paths(self) -> Dict[str, Path]:
        if self._guppy_paths is None:
            self._guppy_paths = {}
            self._guppy_paths['output_paths'] = self._get_output_paths()
            self._guppy_paths['stores_lists_paths'] = self._get_stores_list_paths()
        return self._guppy_paths
