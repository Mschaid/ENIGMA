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
        else:
            return False

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

    def __init__(self, path: Path, meta_data: ExperimentMetadata, path_to_save_figs: Path = None):
        self.meta_data = meta_data(path)
        self._data_frame = None
        self.path_to_save_figs = path_to_save_figs

    @abstractmethod
    def read_and_clean_data(self):
        ...

    @abstractmethod
    def plot_data(self, save: bool = False):
        ...

    def organize_data(self):

        df = pd.read_parquet(self.meta_data.path)
        frame = df.assign(data_cat=self.meta_data.data_category,
                          with_day=self.meta_data.has_day,
                          is_net=self.meta_data.is_net)

        return frame


class MetricAnalyzer(ElasticNetAnalyzer):
    def __init__(self, path, meta_data, path_to_save_figs: Path = None):
        super().__init__(path, meta_data, path_to_save_figs)

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

    def plot_data(self, save: bool = False):

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

        if save:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(self.path_to_save_figs /
                        f"{self.meta_data.full_name}.svg", dpi=300, transparent=True)


class FeatureImportanceAnalyzer(ElasticNetAnalyzer):
    def __init__(self, path, meta_data, path_to_save_figs: Path = None):
        super().__init__(path, meta_data, path_to_save_figs=path_to_save_figs)

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
        label_dict = {
            'day': 'Day',
            'signal_trapz_DA_cue': 'Total DA AUC - Cue',
            'signal_trapz_DA_avoid': 'Total DA AUC - Avoid',
            'signal_min_DA_shock': 'Min DA - Shock',
            'signal_min_DA_escape': 'Min DA - Escape',
            'signal_min_DA_cue': 'Min DA - Cue',
            'signal_min_DA_avoid': 'Min DA - Avoid',
            'signal_max_DA_shock': 'Max DA - Shock',
            'signal_max_DA_escape': 'Max DA - Escape',
            'signal_max_DA_cue': 'Max DA - Cue',
            'pos_signal_trapz_DA_shock': 'Postive DA AUC - Shock',
            'pos_signal_trapz_DA_escape': 'Positive DA AUC - Escape',
            'pos_signal_trapz_DA_avoid': 'Positive DA AUC - Avoid',
            'neg_signal_trapz_DA_shock': 'Negative DA AUC - Shock',
            'neg_signal_trapz_DA_escape': 'Negative DA AUC - Escape',
            'neg_signal_trapz_DA_cue': 'Negative DA AUC - Cue',
            'signal_max_DA_avoid': 'Max DA - Avoid',
            'signal_trapz_DA_escape': 'Total DA AUC - Escape',
            'signal_trapz_DA_shock': 'Total DA AUC - Shock',
            'pos_signal_trapz_DA_cue': 'Positive DA AUC - Cue',
            'neg_signal_trapz_DA_avoid': 'Negative DA AUC - Avoid'
        }

        df = self.organize_data()

        clean_df = (df
                    .assign(
                        feature=lambda df_:
                            pd.Categorical(df_.feature.replace(
                                label_dict), ordered=True)
                    )
                    .sort_values(by='feature')
                    )
        self._data_frame = clean_df
        return clean_df

    def plot_data(self, save=False):
        # colors = sns.color_palette("light:dodgerblue", as_cmap=True)

        def _sorted_features():
            mean_importance = self.data_frame.groupby(
                'feature', observed=True)['importance'].mean()
            sorted_features = mean_importance.sort_values(
                ascending=False).index
            return sorted_features

        sorted_features = _sorted_features()

        err_kws = {'linewidth': 1,
                   }

        sns.barplot(x='importance',
                    y='feature',
                    data=self.data_frame,
                    errorbar='se',
                    orient='h',
                    capsize=.2,
                    palette='Blues_r',
                    legend=False,
                    order=sorted_features,
                    err_kws=err_kws
                    )
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        plt.title('Feature Importance')

        if save:
            file_name = self.path_to_save_figs / \
                f"{self.meta_data.full_name}.svg"
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(file_name, dpi=300, transparent=True)
