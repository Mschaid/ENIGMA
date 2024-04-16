from typing import Generator
import pretty_errors
import pytest
from unittest.mock import Mock, patch, mock_open, PropertyMock
from pathlib import Path
from src.data_processing.pipelines import AAMetaDataFetcher

""" example path "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/A2A-Cre/06.02.22_Day1/309-910_911-220602-140203 copy")"""


class TestAAMetaDataFetcher:

    test_path_1 = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/D1-Cre/06.02.22_Day1/309-910_911-220602-140203/309-910_911-220602-140203_output_1"
    test_path_2 = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross/A2A-Cre/06-02-22_Day2/309-910_911-220602-140203/309-910_911-220602-140203_output_1 copy"
    test_path_save = "not/a/real/path"

    def setup_method(self):
        self.fetcher_1 = AAMetaDataFetcher(
            path=Path(self.test_path_1), path_to_save=(self.test_path_save))
        self.fetcher_2 = AAMetaDataFetcher(
            Path(self.test_path_2), path_to_save=self.test_path_save)

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

    @patch("pathlib.Path.glob")
    def test_fetch_hdf5_path(self, mock_glob):
        mock_path = Path("root_path/sub_path")
        mock_1 = mock_path / "test.hdf5"
        mock_2 = mock_path / "test2.hdf5"
        mock_glob.return_value = [mock_1, mock_2]
        fetcher = AAMetaDataFetcher(mock_path, Path(self.test_path_save))

        keyword = 'test'
        # act
        result = fetcher._fetch_hdf5_paths(keyword)
        result_list = list(result)

        # assert
        assert isinstance(result, Generator)
        assert isinstance(result_list, list)
        assert len(result_list) == 2

    def test_metadata(self):
        meta_data = self.fetcher_1.metadata
        assert isinstance(meta_data, dict)

    def test_save_metadata_to_yaml(self):
        path_to_save = Path(self.test_path_save)
        with patch("builtins.open", mock_open()) as mock_file:
            self.fetcher_1.save_metadata_to_yaml()
            mock_file.assert_called_once_with(
                path_to_save/"cage_309_mouse_910_day_1_metadata.yaml", "w")
