import numpy as np
import pandas as pd
import polars as pl
import h5py

import yaml
from typing import List, Dict, Union, Tuple, Any, NewType
from src.data_processing.processors.FileScraper import FileScraper
from src.data_processing.processors.guppy_processors.experimental_metadata import ExperimentMetaData
from pathlib import Path

Event = NewType('Event', str)
EventToAlign = NewType('EventToAlign', str)


class DataPreprocessor:

    """
    This class is used to load and format data from the guppy experiment from the individual experiments.

    """

    def __init__(self, metadata: ExperimentMetaData):
        self.metadata = metadata
        self._processed_data_df = None

    @property
    def processed_data_df(self)->pd.DataFrame:
        """ returns the processed data as a pandas dataframe"""
        return self._processed_data_df

    def _load_hdf5_file_to_numpy(self, file_path: Path, keyword='timestamps')->np.ndarray:
        """ loads the hdf5 file into a numpy array"""
        time_stamps = h5py.File(file_path, "r").get(keyword)
        return np.array(time_stamps)

    def _create_dict_from_config(self, config_key) -> Dict[str, np.ndarray]:
        """ creates a dictionary of the behavior files from the config file. 
        The keys are the names of the events and the values are the numpy arrays of the timestamps.
        
        Returns
        -------
        Dict[str, np.ndarray]
    
        """
        
        config_dict = {f.stem: self._load_hdf5_file_to_numpy(
            f) for f in self.metadata.behavior_files}

        for stem, event in self.metadata.config_data[config_key].items():
            if stem in config_dict.keys():
                config_dict[event] = config_dict[stem]
                config_dict.pop(stem)
        return config_dict

    def _pad_array_with_nan(self, array: np.array, length: int)->np.array:
        """ pads the array with nan values to the specified length """
        new_arr = np.full(length, np.nan)
        new_arr[:array.shape[0]] = array
        return new_arr

    def generate_df_from_config_dict(self, config_key) -> pd.DataFrame:
        """ generates a dataframe from the config dictionary. 
        The dataframe is padded with nan values to the length of the longest array."""
        
        config_dict = self._create_dict_from_config(config_key=config_key)
        max_length = max([v.shape[0] for v in config_dict.values()])
        padded_behavior_dict = {k: self._pad_array_with_nan(
            v, max_length) for k, v in config_dict.items()}

        return pd.DataFrame(padded_behavior_dict)

    def _align_events(self, df, events: Tuple[Event, EventToAlign])->Dict[str, pd.DataFrame]:
        """ aligns the events to the specified event.
        Returns 
        -------
        Dict[str, pd.DataFrame]
        """
        # unpacks the tuple
        event, event_to_align = events
        # gets the array of the events
        event_array = df[event][np.where(df[event] != 0)[0]].dropna()
        event_to_align_array = df[event_to_align].dropna().to_numpy()

        # creates a dictionary of the aligned events by iterating through the event array
        align_events_dict = {}
        for i in range(event_array.shape[0]):
            arr = np.array(event_to_align_array - event_array[i])
            align_events_dict[i] = arr
            event_dict = {
                k: pd.Series(v) for k, v in align_events_dict.items()
            }
            new_df = pd.DataFrame(event_dict)

        return {
            f"{event_to_align}_aligned_to_{event}": new_df
        }

    def _calculate_mean_event_frequency(self, data: Dict[str, pd.DataFrame], time_window: Tuple[int, int])-> Dict[str, np.ndarray]:
        """ calculate frequency of events around specifc EPOCH and returns a dictionary of the mean frequencies of the events.
        
        Returns  
        -------
        Dict[str, np.ndarray]
        """
        # time is in seconds
        mean_frequencies = {}
        for key, df in data.items():

            arr = df[(df > time_window[0]) & (
                df < time_window[1])].to_numpy()  #
            arr = arr[np.logical_not(np.isnan(arr))]
            freq = np.histogram(arr, bins=155)[0] / len(df.columns)
            freq_convert = freq * 5
            mean_frequencies[key] = freq_convert

        return mean_frequencies

    def batch_calculate_mean_event_frequency(self, data: Dict[str, pd.DataFrame], time_window: Tuple[int, int], *events:Tuple[Event, EventToAlign]):
        """ calculates the mean event frequency for a batch of events.
        * args are the events to be calculated.
        
        Returns
        -------
        Dict[str, np.ndarray]
        """
        results = {}
        for event in events:
            aligned_events = self._align_events(data, event)
            results.update(self._calculate_mean_event_frequency(
                aligned_events, time_window))

        return results

    def _format_meta_df(self)->pd.DataFrame:
        """ formats the metadata dictionary into a pandas dataframe"""
        data = self.metadata.data
        df = (
            pd.DataFrame(data, index=[0])
            .assign(Subject=lambda df_: df_.Subject.astype('int64'),
                    User=lambda df_: df_.User.astype('category'),
                    Date=lambda df_: df_.Date.astype('datetime64[ns]'),
                    Time=lambda df_: df_.Time.astype('datetime64[ns]')
                    )
            .rename(columns=lambda c: c.lower())
        )
        return df

    def aggreate_processed_results(self, data: Dict[str, np.ndarray], return_df: bool = True, save=True)->Union[None, pd.DataFrame]:
        """ aggregates the processed results into a pandas dataframe and saves it as a parquet file.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            dictionary of the processed results
        return_df : bool, optional
            returns the dataframe if true, by default True
        save : bool, optional
            saves the dataframe as a parquet file, by default True
            file is saved into the experiment directory.
            
        Returns
        -------
        Union[None, pd.DataFrame]
            returns the dataframe if return_df is True
        """
        
        meta_df = self._format_meta_df()
        data_df = pd.DataFrame(data)
        joined_df = data_df.join(meta_df, how='outer').fillna(method='ffill')

        if not self._processed_data_df:
            self._processed_data_df = joined_df
        if save:

            self.processed_data_df.to_parquet(
                self.metadata.main_path / f'{self.metadata.experiment_id}_processed_data.parquet')
        if return_df:
            return self.processed_data_df
        else:
            return
