import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List


class SequenceProcessor:
    """Class for processing sequence data for LSTM or similar models.

    Attributes
    ----------
    data : pd.DataFrame
        The input data to be processed
    subject_ids : List[str]
        A list of subject IDs extracted from the column names that match the specified subject prefix
    batches : List[pd.DataFrame]
        A list of dataframes, where each dataframe contains the rows that correspond to a single subject ID
    feature_batches : List[pd.DataFrame]
        A list of feature dataframes for each batch
    target_batches : List[pd.Series]
        A list of target dataframes for each batch
    train_batches_X : List[pd.DataFrame]
        A list of feature dataframes for the training batches
    train_batches_y : List[pd.Series]
        A list of target dataframes for the training batches
    train_subject_ids : List[str]
        A list of subject IDs for the training batches
    val_batches_X : List[pd.DataFrame]
        A list of feature dataframes for the validation batches
    val_batches_y : List[pd.Series]
        A list of target dataframes for the validation batches
    val_subject_ids : List[str]
        A list of subject IDs for the validation batches
    test_batches_X : List[pd.DataFrame]
        A list of feature dataframes for the testing batches
    test_batches_y : List[pd.Series]
        A list of target dataframes for the testing batches
    test_subject_ids : List[str]
        A list of subject IDs for the testing batches

    Methods
    -------
    encode_cyclic_time()
        Encodes the time column as cyclic features
    batch_by_subject(subject_prefix)
        Splits the data into batches by subject ID
    pad_batches(value)
        Pads the batches in the dataset with a specified value
    split_training_val_test_batches(target=None, train_ratio=0.7, validation_ratio=0.15)
        Splits the batches into training, validation, and testing batches by subject
    """

    def __init__(self, data):
        self._data = data
        self.data = data
        self.subject_ids: List[str] = None
        self.batches: List[pd.DataFrame] = None
        self.feature_batches: List[pd.DataFrame] = None
        self.target_batches: List[pd.DataFrame] = None

        # training split
        self.train_batches_X: List[pd.DataFrame] = None
        self.train_batches_y: List[pd.DataFrame] = None
        self.train_subject_ids: List[pd.DataFrame] = None

        # validation split
        self.val_batches_X: List[pd.DataFrame] = None
        self.val_batches_y: List[pd.DataFrame] = None
        self.val_subject_ids: List[pd.DataFrame] = None

        # testing split
        self.test_batches_X: List[pd.DataFrame] = None
        self.test_batches_y: List[pd.DataFrame] = None
        self.test_subject_ids: List[pd.DataFrame] = None

    def query_sensor_and_sort_time_subject(self, sensor):
        """
        Filters and sorts the data

        Parameters
        ----------
        query : str
            The query to filter the data
        sort_by : List[str]
            The list of columns to sort the data by
        """
        sensor_cols = [col for col in self._data.columns if "sensor_" in col]
        mouse_cols = [col for col in self._data.columns if "mouse_id_" in col]

        self.data = self._data.query(
            f'senor_{sensor}==1').sort_values(by=sort_by)
        return self

    def encode_cyclic_time(self):
        max_time = self.data.time.max()
        self.data = (
            self.data
            .assign(
                time_max_norm=lambda df_: (df_.time / max_time) * (2 * np.pi),
                time_cos=lambda df_: np.cos(df_.time_max_norm),
                time_sin=lambda df_: np.sin(df_.time_max_norm)
            )
        )
        return self

    def batch_by_subject(self, subject_prefix):
        """
        Splits the data into batches by subject ID and stores them in the following attributes:
        - subject_ids: a list of subject IDs extracted from the column names that match the specified subject prefix
        - batches: a list of dataframes, where each dataframe contains the rows that correspond to a single subject ID

        Parameters
        ----------
        subject_prefix : str
            The prefix of the column names that contain the subject IDs
        """
        # get all cols with mouse_id
        def query_subject(subject_id):
            return self.data.query(f'{subject_id} == 1').drop(columns=self.subject_ids).reset_index(drop=True)

        # store subject_ids
        self.subject_ids = [
            col for col in self.data.columns if subject_prefix in col]
        # split data into batches
        self.batches = [query_subject(subject_id)
                        for subject_id in self.subject_ids]
        return self

    def pad_batches(self, value):
        """
        Pads the batches in the dataset with a specified value 

        Args:
            value: The value to use for padding.

        Returns:
            The updated dataset object.
        """
        max_length = max([batch.shape[0] for batch in self.batches])
        num_batches = len(self.batches)

        def back_to_df(batch):
            return pd.DataFrame(batch, columns=self.batches[0].columns)

        paded_batches = tf.keras.utils.pad_sequences(
            sequences=self.batches,
            maxlen=max_length,
            dtype='float32',
            padding='post',
            value=value)

        self.batches = [back_to_df(batch) for batch in paded_batches]
        return self

    def split_training_val_test_batches(self, target: str = None, train_ratio: float = 0.7, validation_ratio: float = 0.15):
        """
        Splits the batches into training, validation, and testing batches by subject and stores them in the following attributes:
        - feature_batches: a list of feature dataframes for each batch
        - target_batches: a list of target dataframes for each batch
        - train_batches_X: a list of feature dataframes for the training batches
        - train_batches_y: a list of target dataframes for the training batches
        - train_subject_ids: a list of subject IDs for the training batches
        - val_batches_X: a list of feature dataframes for the validation batches
        - val_batches_y: a list of target dataframes for the validation batches
        - val_subject_ids: a list of subject IDs for the validation batches
        - test_batches_X: a list of feature dataframes for the testing batches
        - test_batches_y: a list of target dataframes for the testing batches
        - test_subject_ids: a list of subject IDs for the testing batches

        Parameters
        ----------
        target : str, optional
            The name of the target column in the batches, by default None
        train_ratio : float, optional
            The ratio of batches to use for training, by default 0.7
        validation_ratio : float, optional
            The ratio of batches to use for validation, by default 0.15
        """
        # calculate indexes for training, validation and testing split
        idx = len(self.batches)
        train_idx = int(idx * train_ratio)
        val_idx = int(idx * validation_ratio) + train_idx

        # store features and targets batches

        self.feature_batches = [df.drop(columns=target) for df in self.batches]
        self.target_batches = [df[target] for df in self.batches]

        # training split
        self.train_batches_X = self.feature_batches[:train_idx]
        self.train_batches_y = self.target_batches[:train_idx]
        self.train_subject_ids = self.subject_ids[:train_idx]

        # validation split
        self.val_batches_X = self.feature_batches[train_idx:val_idx]
        self.val_batches_y = self.target_batches[train_idx:val_idx]
        self.val_subject_ids = self.subject_ids[train_idx:val_idx]

        # testing split
        self.test_batches_X = self.feature_batches[val_idx:]
        self.test_batches_y = self.target_batches[val_idx:]
        self.test_subject_ids = self.subject_ids[val_idx:]

        return self

    def reshape_batches(self):
        def batches_to_np(batches):
            return np.stack([batch.to_numpy() for batch in batches])

        batch_attr_name = [name for name in dir(self) if '_batches_' in name]

        for name in batch_attr_name:
            arr = getattr(self, name)
            setattr(self, name, batches_to_np(arr))

        return self
