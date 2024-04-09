import pretty_errors
import pytest
from unittest.mock import Mock, patch, mock_open, PropertyMock
from pathlib import Path
from src.data_processing.pipelines import AAMetaDataFetcher

""" example path "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/A2A-Cre/06.02.22_Day1/309-910_911-220602-140203 copy")"""


class TestAAMetaDataFetcher:

    test_path_1 = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/D1-Cre/06.02.22_Day1/309-910_911-220602-140203/309-910_911-220602-140203_output_1"
    test_path_2 = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/A2A-Cre/06-02-22_Day2/309-910_911-220602-140203/309-910_911-220602-140203_output_1 copy"

    def setup_method(self):
        self.fetcher_1 = AAMetaDataFetcher(Path(self.test_path_1))
        self.fetcher_2 = AAMetaDataFetcher(Path(self.test_path_2))

    def test_path(self):
        assert isinstance(self.fetcher_1.path, Path)
        assert isinstance(self.fetcher_2.path, Path)

    def test_fetch_day(self):
        assert self.fetcher_1.fetch_day() == 1
        assert self.fetcher_2.fetch_day() == 2

    def test_fetch_cage(self):
        assert self.fetcher_1.fetch_cage() == 309

    def test_mouse_id_no_copy(self):
        assert self.fetcher_1.fetch_mouse_id() == 910

    def test_mouse_id_with_copy(self):
        assert self.fetcher_2.fetch_mouse_id() == 911

    def test_fetch_D1(self):
        assert self.fetcher_1.fetch_D1()
        assert not self.fetcher_2.fetch_D1()

    def test_fetch_D2(self):
        assert not self.fetcher_1.fetch_D2()
        assert self.fetcher_2.fetch_D2()

    def test_DA(self):
        assert self.fetcher_1.fetch_DA()
        assert self.fetcher_2.fetch_DA()

    def test_metadata(self):
        meta_data = self.fetcher_1.metadata
        assert isinstance(meta_data, dict)

    def test_save_metadata_to_yaml(self):
        with patch("builtins.open", mock_open()) as mock_file:
            self.fetcher_1.save_metadata_to_yaml()
        mock_file.assert_called_once_with(
            self.fetcher_1.path/"metadata.yaml", "w")
