
import os
import numpy as np
import pandas as pd
import json


from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utilities.pandas_helpers import filter_columns_by_search, flatten_dataframe


"""
A pipeline for processing and transforming data for classification models.

Methods:
- read_raw_data(): Reads raw data from a file and stores it in the 'raw_data' attribute.
- calculate_max_min_signal(): Calculates the maximum and minimum signal values for each combination of features and stores the processed data in the 'processed_data' attribute.
- split_data(): Splits the data into train and test sets, optionally stratifying by a specified column. It also saves the subject IDs to a JSON file and the datasets to parquet files if specified.
- transform_data(): Transforms the data by converting the target to numeric encoding and applying pipelines to the input data for modeling. The transformed data is stored in the 'X_train', 'X_dev', and 'X_test' attributes.

Attributes:
- path_to_data: The path to the raw data file.
- raw_data: The raw data read from the file.
- processed_data: The processed data with maximum and minimum signal values.
- X_train, X_dev, X_test: The transformed input data for training, development, and testing.
- y_train, y_dev, y_test: The transformed target data for training, development, and testing.
"""


class ClassifierPipe:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    # read raw data

    def read_raw_data(self, features_to_drop=None):
        """read raw data from path_to_data

        Returns
        -------
            ClassifierPipe object
        """
        if features_to_drop is not None:
            self.raw_data = pd.read_parquet(
                self.path_to_data).drop(columns=features_to_drop)
        else:
            self.raw_data = pd.read_parquet(self.path_to_data)
        return self
    # reduce signal to max and min

    def calculate_max_min_signal(self, cols_to_drop=[]):
        """
        Calculate the maximum and minimum signal values for each combination of mouse, event, action, sensor, sex, day, trial_count, and trial in the raw data.

        Parameters:
            calculate_auc : bool, optional
                If True, calculate the area under the curve (AUC) for the signal values. Default is False.
            cols_to_drop : list, optional
                A list of column names to drop from the processed data. Default is an empty list.

        Returns:
            self : ClassifierPipe
                The modified ClassifierPipe object with the processed data.

        Examples:
            # Example 
            >>> pipe = ClassifierPipe(PATH_TO_DATA)
            >>> pipe.read_raw_data()
            >>> pipe.calculate_max_min_signal(calculate_auc=True, cols_to_drop=["A", "B"])

            """

        events = filter_columns_by_search(self.raw_data, 'event')
        actions = filter_columns_by_search(self.raw_data, 'action')
        mouse = filter_columns_by_search(self.raw_data, 'mouse')
        sensors = filter_columns_by_search(self.raw_data, 'sensor')
        sex = filter_columns_by_search(self.raw_data, 'sex')

        self.processed_data = (
            self.raw_data
            .query("time > -0.10 and time < 10.1")
            .assign(
                neg_signal=lambda df_: np.where(df_.signal < 0, df_.signal, 0),
                pos_signal=lambda df_: np.where(df_.signal > 0, df_.signal, 0)
            )
            .groupby(by=mouse+events+actions+sensors+sex+['day', 'trial_count', 'trial'], as_index=False).agg({"signal": ["max", "min", np.trapz],                                                                                                           "pos_signal": np.trapz})
            .pipe(flatten_dataframe)
            .rename(columns=lambda c: c.strip("_"))
            .drop(columns='index')
            .dropna()
        )
        self.processed_data = self.processed_data.drop(columns=cols_to_drop)

        return self

    def calculate_percent_avoid(self):

        # calculate max trials
        max_trials = self.processed_data.groupby(["mouse_id", "day"], as_index=False)[
            "trial"].max().rename(columns={"trial": "max_trial"})

        # merge max_trials
        merged_df = self.processed_data.merge(
            max_trials, on=["mouse_id", "day"], how="right")

        num_avoids = (merged_df
                      .query("event == 'cue' & sensor=='DA' & action=='avoid'")
                      .groupby(by=["mouse_id", "day", "action"], as_index=False).size().rename(columns={"size": "num_avoids"})
                      .drop(columns="action")
                      )

        # calualte ratio avoid and update processed data
        self.processed_data = (
            self.processed_data
            .merge(num_avoids, on=["mouse_id", "day"], how="right")
            .merge(max_trials, on=["mouse_id", "day"], how="left")
            .assign(ratio_avoid=lambda df_: (df_.num_avoids/df_.max_trial))
        )
        return self

    def drop_features(self, cols_to_drop):
        self.processed_data = self.processed_data.drop(columns=cols_to_drop)
        if hasattr(self, 'X_train'):
            self.X_train = self.X_train.drop(columns=cols_to_drop)
            self.X_dev = self.X_dev.drop(columns=cols_to_drop)
            self.X_test = self.X_test.drop(columns=cols_to_drop)
        return self

    # split data into train and test and save subject ids to json

    def split_data(self,
                   test_size=0.2,
                   test_dev_size=0.5,
                   split_group=None,
                   stratify_group=None,
                   target=None,
                   processed_data=True,
                   save_subject_ids=True,
                   save_datasets=False,
                   path_to_save=None,
                   load_subject_ids=False,
                   subject_ids_path=None
                   ):
        """
        Calculate the maximum and minimum signal values for each combination of event, action, mouse, sensor, sex, day, and trial count.

        This function filters the columns of the raw data based on specific search terms ('event', 'action', 'mouse', 'sensor', 'sex') and assigns the filtered columns to variables. Then, it performs the following operations on the raw data:
        - Groups the data by the filtered columns and additional columns ('day', 'trial_count')
        - Calculates the maximum and minimum values of the 'signal' column for each group
        - Flattens the resulting dataframe
        - Renames the columns by removing the '_' character
        - Drops the 'index' column

        Parameters
        ----------
        None

        Returns
        -------
        self: MyClass
            The modified object with the processed data.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.calculate_max_min_signal()
        """

        if processed_data is True:
            df = self.processed_data
        else:
            df = self.raw_data

        if load_subject_ids is True:
            with open(subject_ids_path) as f:
                self.subject_category = json.load(f)

            def query_mouse_id(self, category):
                mice = self.subject_category[category]
                query = " or ".join([f"mouse_id == {mouse}" for mouse in mice])
                return query

            train_query, dev_query, test_query = query_mouse_id(
                self, 'training'), query_mouse_id(self, 'dev'), query_mouse_id(self, 'test')
            training_set, dev_set, test_set = self.processed_data.query(
                train_query), self.processed_data.query(dev_query), self.processed_data.query(test_query)

            self.X_train, self.X_dev, self.X_test = training_set.drop(
                columns=target), dev_set.drop(columns=target), test_set.drop(columns=target)
            self.y_train, self.y_dev, self.y_test = training_set[
                target], dev_set[target], test_set[target]
            return self

        X = df.drop(columns=target)
        y = df[target]

        def train_test_split_by_group_stratify(X, y, group_col, stratify_col, test_size, random_state=42):

            group_splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state)
            group_train_idx, group_test_idx = next(
                group_splitter.split(X, y, groups=X[group_col]))

            strat_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.5, random_state=random_state)
            strat_train_idx, _ = next(strat_splitter.split(
                X.iloc[group_train_idx], X.iloc[group_train_idx][stratify_col]))

            train_idx = group_train_idx[strat_train_idx]
            test_idx = group_test_idx

            x_train = X.iloc[train_idx]
            x_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            return x_train, x_test, y_train, y_test

        self.X_train, X_temp, self.y_train, y_temp = train_test_split_by_group_stratify(
            X, y, group_col=split_group, stratify_col=stratify_group, test_size=test_size)
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split_by_group_stratify(
            X_temp, y_temp, group_col=split_group, stratify_col=stratify_group, test_size=test_dev_size)

        # save subject ids to json

        if save_subject_ids is True:

            train_subjects = self.X_train['mouse_id'].unique().tolist()
            dev_subjects = self.X_dev['mouse_id'].unique().tolist()
            test_subjects = self.X_test['mouse_id'].unique().tolist()
            self.subject_category = {
                "training": train_subjects,
                "dev": dev_subjects,
                "test": test_subjects

            }
            with open(os.path.join(path_to_save, 'subjects.json'), 'w') as f:
                json.dump(self.subject_category, f)

        if save_datasets is True:
            # save datasets to parquet.gzip
            self.X_train.to_parquet(os.path.join(
                path_to_save, 'X_train.parquet.gzip'), compression='gzip')
            self.X_dev.to_parquet(os.path.join(
                path_to_save, 'X_dev.parquet.gzip'), compression='gzip')
            self.X_test.to_parquet(os.path.join(
                path_to_save, 'X_test.parquet.gzip'), compression='gzip')
            self.y_train.to_parquet(os.path.join(
                path_to_save, 'y_train.parquet.gzip'), compression='gzip')
            self.y_dev.to_parquet(os.path.join(
                path_to_save, 'y_dev.parquet.gzip'), compression='gzip')
            self.y_test.to_parquet(os.path.join(
                path_to_save, 'y_test.parquet.gzip'), compression='gzip')

        # if loading from stored subject ids

    # create pipeline

    def transform_data(self, numeric_target_dict=None):
        """
        Transforms the data by converting the target to numeric encoding. 

        Parameters
        ----------
        numeric_target_dict : dict, optional
            A dictionary that maps the target values to their corresponding 
            numeric encodings. Defaults to None.

        Returns
        -------
        self : object
            The modified instance of the class.

        Description
        -----------
        This function converts the target values in the training, development, 
        and testing sets to their corresponding numeric encoding using the 
        provided dictionary `numeric_target_dict`. If `numeric_target_dict` 
        is not provided, no conversion is performed.

        After converting the target values, this function applies a pipeline 
        to the input data `X` to transform it for modeling. The pipeline 
        consists of two steps: 

        1. Numeric Pipeline: This pipeline handles the numeric features in `X`. 
           It performs the following operations:
            - Imputation: Missing values in the numeric features are imputed 
              using the mean value of the corresponding feature in the training set.
            - Scaling: The imputed numeric features are scaled using the 
              StandardScaler.

        2. Categorical Pipeline: This pipeline handles the categorical features 
           in `X`. It performs the following operations:
            - Imputation: Missing values in the categorical features are imputed 
              using the most frequent value of the corresponding feature in the 
              training set.
            - One-Hot Encoding: The imputed categorical features are one-hot 
              encoded using the OneHotEncoder with 'ignore' strategy for 
              handling unknown categories.

        Finally, a ColumnTransformer is used to apply the numeric and categorical 
        pipelines to the respective features in `X`. The transformed data is then 
        stored in the instance variables `X_train`, `X_dev`, and `X_test`.

        Returns the modified instance of the class.
        """

        # covert target to numeric encoding

        if numeric_target_dict is not None:

            self.y_train = self.y_train.replace(numeric_target_dict)
            self.y_dev = self.y_dev.replace(numeric_target_dict)
            self.y_test = self.y_test.replace(numeric_target_dict)

        # pipeline for x data
        numeric_features = self.X_train.select_dtypes(
            include='number').columns.tolist()
        categorical_features = self.X_train.select_dtypes(
            include=['object', 'category']).columns.tolist()

        numeric_pipeline = Pipeline([

            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

        categorical_pipeline = Pipeline([

            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ])
        self.processor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ]

        )

        self.processor.fit(self.X_train)
        self.X_train = self.processor.transform(self.X_train)
        self.X_dev = self.processor.transform(self.X_dev)
        self.X_test = self.processor.transform(self.X_test)
        return self
