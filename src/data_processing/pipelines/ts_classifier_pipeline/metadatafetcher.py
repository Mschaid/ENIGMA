import yaml
from pathlib import Path


class MetaDataFetcher:
    """
        class used to extract metadata from a given path.

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
        self._day = None
        self._cage = None
        self._mouse_id = None
        self._D1 = None
        self._D2 = None
        self._DA = True

    @property
    def day(self) -> int:
        """ extracts the day from the file name and returns it as an int."""
        if not self._day:
            parent_name = self.path.parent.name
            day_string = parent_name.split("_")[1]
            self._day = int(day_string[-1])
        return self._day

    @property
    def cage(self) -> int:
        """ extracts the cage number from the file name and returns it as an int."""
        if not self._cage:
            parent_name = self.path.name
            cage_string = parent_name.split("-")[0]
            self._cage = int(cage_string)
        return self._cage

    @property
    def mouse_id(self) -> int:
        """extracts the mouse ID from the file name and returns it as an int."""
        if not self._mouse_id:
            name_split = self.path.name.split("-")
            ids = name_split[1].split("_")
            has_copy = "copy" in self.path.name
            if not has_copy:
                self._mouse_id = int(ids[0])
            else:
                self._mouse_id = int(ids[1])
        return self._mouse_id

    @property
    def D1(self) -> bool:
        """ returns True if "D1" is in the file name, False otherwise."""
        if not self._D1:
            self._D1 = "D1" in self.path.name
        return self._D1

    @property
    def D2(self) -> bool:
        """ returns True if "A2A" is in the file name, False otherwise."""
        if not self._D2:
            self._D2 = "A2A" in self.path.name
        return self._D2

    @property
    def DA(self) -> bool:
        """ always returns True, given that this dataset always records dopamine"""
        return self._DA

    def metadata(self):
        """ returns a dictionary containing the metadata extracted from the file name."""
        return {
            "day": self.day,
            "cage": self.cage,
            "mouse_id": self.mouse_id,
            "D1": self.D1,
            "D2": self.D2,
            'DA': self.DA
        }

    def save_metadata_to_yaml(self):
        """ saves the metadata to a yaml file in the directory pointed to by the path attribute)."""
        with open(self.path/"metadata.yaml", "w") as f:
            yaml.dump(self.metadata(), f)
