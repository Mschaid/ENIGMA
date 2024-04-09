
import logging
import pretty_errors
import yaml
from pathlib import Path

from typing import List, Protocol


class MetaDataFetcher(Protocol):
    def __init__(self, path: Path):
        """ takes output guppy path as Path object"""
        ...

    def metadata(self) -> dict:
        """ returns a dictionary containing the metadata extracted from the directory parameters, as well as any filepaths."""
        ...

    def save_metadata_to_yaml(self) -> None:
        """ saves the metadata to a yaml file in the directory pointed to by the path attribute."""
        ...

# simple methods used by the MetaDataFetcher class implementation and factory function


def directory_finder(main_path: Path, directory_keyword: str) -> List[Path]:
    paths_found = main_path.glob(f"**/*{directory_keyword}*")
    return [path for path in paths_found if path.is_dir()]


def meta_data_factory(path: Path, fetcher: MetaDataFetcher) -> MetaDataFetcher:
    fetcher = MetaDataFetcher(path)
    return fetcher


class AAMetaDataFetcher(MetaDataFetcher):
    """
        class used to extract metadata from active avoidance experiments from a given path.

    Attributes
    ----------
    path : Path
        a Path object that points to the file from which metadata is extracted
    day : int
        the day extracted from the file name (default is None)
    cage : int
        the cage number extracted from the file name (default is None)
    mouse_id : int
        the mouse ID extracted from the file name (default is None)
    D1 : bool
        a flag indicating if "D1" is in the file name (default is None)
    D2 : bool
        a flag indicating if "A2A" is in the file name (default is None)
    dDA : bool
        a flag always set to True


    """

    def __init__(self, path: Path):
        self.path = path
        self._metadata: dict = None

    def fetch_day(self) -> int:
        """ extracts the day from the file name and returns it as an int."""
        parent_name = self.path.parents[1].name
        day_string = parent_name.split("_")[1]
        day = int(day_string[-1])
        return day

    def fetch_cage(self) -> int:
        """ extracts the cage number from the file name and returns it as an int."""
        parent_name = self.path.name
        cage_string = parent_name.split("-")[0]
        cage = int(cage_string)
        return cage

    def fetch_mouse_id(self) -> int:
        """extracts the mouse ID from the file name and returns it as an int."""
        name_split = self.path.name.split("-")
        ids = name_split[1].split("_")
        has_copy = "copy" in self.path.name
        if not has_copy:
            mouse_id = int(ids[0])
        else:
            mouse_id = int(ids[1])
        return mouse_id

    def fetch_D1(self) -> bool:
        """ returns True if "D1" is in the file name, False otherwise."""
        is_D1 = "D1" in self.path.as_posix()
        return is_D1

    def fetch_D2(self) -> bool:
        """ returns True if "A2A" is in the file name, False otherwise."""
        is_D2 = "A2A" in self.path.as_posix()
        return is_D2

    def fetch_DA(self) -> bool:
        """ always returns True, given that this dataset always records dopamine"""
        return True

    def fetch_full_z_scored_recordings(self):
        z_scored_paths = []
        for file in self.path.glob("*z_score*.hdf5"):
            posix_path = file.as_posix()
            z_scored_paths.append(posix_path)
        return z_scored_paths

    @property
    def metadata(self):
        """ returns a dictionary containing the metadata extracted from the file name."""
        if self._metadata is None:
            self._metadata = {
                "day": self.fetch_day(),
                "cage": self.fetch_cage(),
                "mouse_id": self.fetch_mouse_id(),
                "D1": self.fetch_D1(),
                "D2": self.fetch_D2(),
                'DA': self.fetch_DA(),
                "full_z_scored_recording_paths": self.fetch_full_z_scored_recordings()
            }
            return self._metadata

    def save_metadata_to_yaml(self):
        """ saves the metadata to a yaml file in the directory pointed to by the path attribute)."""
        with open(self.path/"metadata.yaml", "w") as f:
            yaml.dump(self.metadata, f)
