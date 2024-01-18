import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from abc import ABC, abstractmethod
from functools import lru_cache


class ElasticNetAnalyzer(ABC):

    @abstractmethod
    def read_and_clean_data(self):
        ...

    @abstractmethod
    def plot_data(self):
        ...

    def organize_data(self, path: Path):
        def find_has_day_in_string(exp_cat):
            if re.match('with_day', exp_cat):
                return True
            else:
                return False

        def find_net_in_string(exp_cat):
            if re.match('net', exp_cat):
                return True
            else:
                return False
        data_cat = path.stem
        experiment_cat = path.parent.stem
        group = path.parent.parent.stem
        df = pd.read_parquet(path)
        frame = df.assign(data_cat=data_cat,
                          with_day=find_has_day_in_string(experiment_cat),
                          is_net=find_net_in_string(group))

        return frame


class MetricAnalyzer(ElasticNetAnalyzer):
    def __init__(self, path: Path):
        self.path = path
        self._data_frame = None

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
        df = self.organize_data(self.path)
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
    def __init__(self, path: Path):
        self.path = path
        self._data_frame = None

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

        df = self.organize_data(self.path)

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
