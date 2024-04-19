import pytest

from src.data_processing.pipelines import (GuppyDataFetcher,
                                           filter_metadata_paths_for_keywords,
                                           get_max_value_length,
                                           pad_dict_arrays,
                                           read_hdf5_event_timestamps_data,
                                           read_hdf5_fp_signal_data)


def test_filter_metadata_paths_for_keywords():
    test_metadata = {
        "keyword1_data": ["value1", "value2"],
        "keyword2_data": ["value3", "value4"],
        "filter_key_data": ["value5", "value6"],
        "keyword1_filter_key_data": ["value7", "value8"],
        "keyword2_other_data": ["value9", "value10"],
    }
    filtered_values = filter_metadata_paths_for_keywords(
        'filter_key', metadata=test_metadata)
    assert filtered_values == ['value5', 'value6', 'value7', 'value8']
