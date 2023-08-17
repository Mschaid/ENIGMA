
import os
import numpy as np
import pandas as pd
import json


from sklearn.model_selection import train_test_split
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

    def read_raw_data(self):
        """read raw data from path_to_data

        Returns
        -------
            ClassifierPipe object
        """

        self.raw_data = pd.read_parquet(self.path_to_data)
        return self
    # reduce signal to max and min

    def calculate_max_min_signal(self):
        """
        Calculate the maximum and minimum signal values for each combination of mouse, event, action, sensor, sex, day, and trial count.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            self (object): The updated object.

        Examples:
            >>> data = DataProcessor()
            >>> data.calculate_max_min_signal()
        """

        events = filter_columns_by_search(self.raw_data, 'event')
        actions = filter_columns_by_search(self.raw_data, 'action')
        mouse = filter_columns_by_search(self.raw_data, 'mouse')
        sensors = filter_columns_by_search(self.raw_data, 'sensor')
        sex = filter_columns_by_search(self.raw_data, 'sex')
        self.processed_data = (
            self.raw_data
            .groupby(by=mouse+events+actions+sensors+sex+['day', 'trial_count'], as_index=False).agg({"signal": ["max", "min"]})
            .pipe(flatten_dataframe)
            .rename(columns=lambda c: c.strip("_"))
            .drop(columns='index')
        )

        return self

    # split data into train and test and save subject ids to json
    def split_data(self,
                   test_size=0.2,
                   test_val_size=0.5,
                   stratify_by=None,
                   target=None,
                   processed_data=True,
                   save_subject_ids=True,
                   save_datasets=False,
                   path_to_save=None
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

        X = df.drop(columns=target)
        y = df[target]
        
        #! something is wrong here, the attributes are not returning split data, all subjects are in every dataset
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=df[stratify_by])
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            X_temp, y_temp, test_size=test_val_size, stratify=X_temp[stratify_by])
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

        return self
    # create pipeline

    def transorm_data(self, numeric_target_dict=None):
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
        processor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ]

        )

        processor.fit(self.X_train)
        self.X_train = processor.transform(self.X_train)
        self.X_dev = processor.transform(self.X_dev)
        self.X_test = processor.transform(self.X_test)
        return self
