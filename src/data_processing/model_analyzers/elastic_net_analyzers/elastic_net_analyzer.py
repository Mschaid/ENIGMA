import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass

from abc import ABC, abstractmethod
from functools import lru_cache


class ExperimentMetadata:
    def __init__(self, path: str):
        self.path = Path(path)

    @property
    def experiment_name(self):
        return self.path.parents[1].name

    @property
    def net_status(self):
        if re.search("net", self.path.as_posix()):
            return "elastic_net"
        else:
            return "baseline"

    @property
    def is_net(self):
        if self.net_status == "elastic_net":
            return True

    @property
    def day_status(self):
        if re.search("with_day", self.path.as_posix()):
            return "with_day"
        else:
            return "without_day"

    @property
    def has_day(self):
        if self.day_status == "with_day":
            return True

    @property
    def data_category(self):
        if re.search("metric", self.path.name):
            return "metrics"
        if re.search("feature", self.path.name):
            return "feature_importance"

    @staticmethod
    def _get_group(path):

        if re.search("da_only", path):
            return "da_only"
        if re.search("da_and_d1", path):
            return "da_and_d1"
        if re.search("da_and_d2", path):
            return "da_and_d2"

    @property
    def group(self):
        return ExperimentMetadata._get_group(self.path.as_posix())

    @property
    def full_name(self):
        name = f"{self.group}_{self.net_status}_{self.day_status}_{self.data_category}"
        return name


class ElasticNetAnalyzer(ABC):

    def __init__(self, path: Path, meta_data: ExperimentMetadata):
        self.meta_data = meta_data(path)
        self._data_frame = None

    @abstractmethod
    def read_and_clean_data(self):
        ...

    @abstractmethod
    def plot_data(self):
        ...

    def organize_data(self):

        df = pd.read_parquet(self.meta_data.path)
        frame = df.assign(data_cat=self.meta_data.data_category,
                          with_day=self.meta_data.has_day,
                          is_net=self.meta_data.is_net)

        return frame


class MetricAnalyzer(ElasticNetAnalyzer):
    def __init__(self, path, meta_data):
        super().__init__(path, meta_data)

    @property
    # @lru_cache(maxsize=None)
    def data_frame(self):
        if self._data_frame is None:
            print('reading and cleaning data')
            self.read_and_clean_data()
        return self._data_frame

    @data_frame.setter
    def data_frame(self, df):
        self.data_frame = df

    def read_and_clean_data(self):
        df = self.organize_data()
        clean_df = (
            df
            .reset_index()
            .rename(columns={'index': 'dataset'})
            .melt(id_vars=["dataset", "data_cat", "with_day", "is_net"],
                  value_vars=["mean_squared_error",
                              "mean_absolute_error", "r2_score"],
                  var_name="metric")
            .assign(dataset=lambda df_: df_['dataset'].str.capitalize(),
                    metric=lambda df_: df_.metric.str.replace(
                        '_', ' ').str.title()
                    )
        )
        self._data_frame = clean_df

    def plot_data(self):

        err_kws = {'linewidth': 1,
                   }

        facet = sns.FacetGrid(self.data_frame, col="metric", sharey=False).map(
            sns.barplot,
            'dataset',
            'value',
            data=self.data_frame,
            order=['Train', 'Dev', 'Test'],
            hue='dataset',
            palette=['white', 'silver', '#56B4E9'],
            width=.9,
            edgecolor='black',
            estimator='mean',
            errorbar='se',
            capsize=.1,
            err_kws=err_kws)
        facet.set_ylabels("Metric Value")
        facet.set_xlabels("Dataset")
        facet.set_titles("{col_name}")


class FeatureImportanceAnalyzer(ElasticNetAnalyzer):
    def __init__(self, path, meta_data):
        super().__init__(path, meta_data)

    @property
    # @lru_cache(maxsize=None)
    def data_frame(self):
        if self._data_frame is None:
            print('reading and cleaning data')
            self.read_and_clean_data()
        return self._data_frame

    @data_frame.setter
    def data_frame(self, df):
        self.data_frame = df

    def read_and_clean_data(self):
        def _sort_features(df):
            mean_importance = df.groupby('feature')[
                'importance'].mean()

            # Sort the features by mean 'importance'
            sorted_features = (mean_importance
                               .sort_values(ascending=False).index.str.replace('_', ' ').str.title())
            return sorted_features

        df = self.organize_data()

        sorted_features = _sort_features(df)

        clean_df = (df
                    .assign(feature=lambda df_:
                            pd.Categorical(df_.feature.str.replace(
                                '_', ' ').str.title(), categories=sorted_features, ordered=True)
                            )
                    )
        self._data_frame = clean_df

    def plot_data(self):
        # colors = sns.color_palette("light:dodgerblue", as_cmap=True)

        err_kws = {'linewidth': 1,
                   }

        sns.barplot(x='importance',
                    y='feature',
                    data=self.data_frame,
                    errorbar='se',
                    orient='h',
                    capsize=.2,
                    palette='Blues_r',
                    hue='feature',
                    err_kws=err_kws)
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
