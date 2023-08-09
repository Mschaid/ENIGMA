import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
import json


class TrainingProcessor:
    def __init__(self, data: pd.DataFrame):

        self._data = data  # orignal data
        self.data = data  # origianl data that is modified by the processor

    def drop_colinear_columns(self, colinearity_cols: List[str] = []):
        self.data = self.data.drop(columns=colinearity_cols)
        return self

    def query_sensor_and_sort_trials_by_subject(self, sensor: str):
        sensor_cols = [col for col in self.data.columns if "sensor_" in col]
        mouse_cols = [col for col in self.data.columns if "mouse_id_" in col]

        self.data = (
            self.data
            .query(f"sensor_{sensor} == 1")
            .drop(columns=sensor_cols)
            .reset_index(drop=True)
            .sort_values(by=mouse_cols + ["trial_count"])
        )
        return self

    def split_train_val_test_by_subject(self, target: str, train_ratio: float = 0.7, validation_ratio: float = 0.15, shuffle=True):
        """
        Splits the data into train, validation and test sets by subject.
        """
        # Get unique subjects

        def query_subjects(subjects: List[str]):
            return pd.concat([self.data.query(f"{subject} == 1") for subject in subjects])
        self.subjects = [col for col in self.data.columns if "mouse_id" in col]

        def split_x_y(data: pd.DataFrame, target):
            return data.drop(columns=target), data[target]

        # Shuffle subjects
        if shuffle:
            np.random.shuffle(self.subjects)
        # Split subjects into train, validation and test sets
        num_subjects = len(self.subjects)
        num_training_subjects = int(num_subjects * train_ratio)
        num_validation_subjects = int(num_subjects * validation_ratio)
        num_test_subjects = num_subjects - num_training_subjects - num_validation_subjects

        self.training_subjects = self.subjects[:num_training_subjects]
        self.validation_subjects = self.subjects[num_training_subjects:
                                                 num_training_subjects + num_validation_subjects]
        self.testing_subjects = self.subjects[-num_test_subjects:]

        self.training_data = query_subjects(self.training_subjects)
        self.validation_data = query_subjects(self.validation_subjects)
        self.testing_data = query_subjects(self.testing_subjects)

        self.train_x, self.train_y = split_x_y(self.training_data, target)
        self.val_x, self.val_y = split_x_y(self.validation_data, target)
        self.test_x, self.test_y = split_x_y(self.testing_data, target)

        return self

    def save_subjects_by_category(self, path):

        path_to_save = os.path.join(path, "subjects_by_category.json")
        subjects_category = {
            "training": self.training_subjects,
            "validation": self.validation_subjects,
            "testing": self.testing_subjects
        }

        with open(path_to_save, "w") as f:
            json.dump(subjects_category, f)

    def load_subjects_by_category(self, path):
        with open(path, "r") as f:
            subjects_category = json.load(f)
        return subjects_category
