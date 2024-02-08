
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance


from typing import Callable, List, Optional, Tuple, Dict, Any, Literal, Protocol
from pathlib import Path


# local imports
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegAnalyzer import XGBRegAnalyzer,  XGBNormRegAnalyzer
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegrResults import XGBRegrResults

from abc import ABC, abstractmethod


class Experimenter(Protocol):
    path: str
    _results: XGBRegrResults
    analyzer: XGBRegAnalyzer
    _experiment_metric_results: pd.DataFrame
    experiment_results: Dict[str, pd.DataFrame]
    analyzer_runs: List[XGBRegAnalyzer]

    @property
    def results(self) -> XGBRegrResults:
        if not self._results:
            self._results = XGBRegrResults(self.path)
        return self._results

    def run_experiment(self, number_of_runs: 10) -> None:
        pass

    def run_permutation_experiment(self, number_of_runs: 10) -> None:
        pass

    def save_results(self, condition_name=''):
        pass


class XGBRegExperimenter:
    def __init__(self, path, analyzer, results):
        self.path = Path(path)
        self.results = results(path)
        self.analyzer = analyzer
        self._experiment_metric_results = None
        self.experiment_results: Dict[str, pd.DataFrame] = {}
        self.analyzer_runs: List[XGBRegAnalyzer] = []

    def run_permutation_experiment(self, number_of_runs, cls_to_drop: List[str] = []):
        analyzer = self.analyzer(self.results)
        analyzer.create_pipeline(cls_to_drop=cls_to_drop)
        X = analyzer.pipeline.X_train
        y = analyzer.pipeline.y_train
        model = analyzer.best_xgb_model.fit(X, y)
        result = permutation_importance(model, X, y, scoring=[
                                        'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'], n_repeats=10)

        return result, analyzer.feature_names

    def run_experiment(self, number_of_runs: 10, cls_to_drop: List[str] = []):
        metric_runs = []
        feature_importance_runs = []
        for _ in range(number_of_runs):
            analyzer = self.analyzer(self.results)
            analyzer.create_pipeline(cls_to_drop=cls_to_drop)
            self.analyzer_runs.append(analyzer)  # keep track of runs
            analyzer.fit_best_xgb_model()
            run_metric_results = pd.DataFrame(analyzer.metrics_results)
            run_feature_importance = pd.DataFrame(
                analyzer.feature_importance_df)

            # keep track of metrics dataframes
            metric_runs.append(run_metric_results)
            # keep track of feature importance dataframes
            feature_importance_runs.append(run_feature_importance)

        compiled_metric_results = pd.concat(metric_runs)
        compiled_feature_importance = pd.concat(feature_importance_runs)

        self.experiment_results.update(
            metric_results=compiled_metric_results)
        self.experiment_results.update(
            feature_importance_results=compiled_feature_importance)

    def save_results(self, condition_name: str):
        path_to_save = self.path / f"{condition_name}_experiment_results"
        path_to_save.mkdir(parents=True, exist_ok=True)
        for key, df in self.experiment_results.items():

            df.to_parquet(f'{path_to_save}/{key}.parquet')


class XGBRegExperimenterFactory:
    def __init__(self, path: Path):
        self.path = path
        self.analyzer = XGBRegAnalyzer
        self.results = XGBRegrResults

    def create_experimenter(self) -> XGBRegExperimenter:
        return XGBRegExperimenter(self.path, self.analyzer, self.results)


class XGBNormRegExperimenter(Experimenter):
    def __init__(self, path, analyzer, results_reader):
        self.path = Path(path)
        self._results = results_reader(path)
        self.analyzer = analyzer
        self._experiment_metric_results = None
        self.experiment_results: Dict[str, pd.DataFrame] = {}
        self.analyzer_runs: List[XGBNormRegAnalyzer] = []

    def run_experiment(self, number_of_runs: 10, cls_to_drop: List[str] = []):
        metric_runs = []
        feature_importance_runs = []

        for _ in range(number_of_runs):
            analyzer = self.analyzer(self.results)
            analyzer.create_pipeline(cls_to_drop=cls_to_drop)
            self.analyzer_runs.append(analyzer)  # keep track of runs
            analyzer.fit_best_xgb_model()
            run_metric_results = pd.DataFrame(analyzer.metrics_results)
            run_feature_importance = pd.DataFrame(
                analyzer.feature_importance_df)

            # keep track of metrics dataframes
            metric_runs.append(run_metric_results)
            # keep track of feature importance dataframes
            feature_importance_runs.append(run_feature_importance)


        compiled_metric_results = pd.concat(metric_runs)
        compiled_feature_importance = pd.concat(feature_importance_runs)

        self.experiment_results.update(
            metric_results=compiled_metric_results)
        self.experiment_results.update(
            feature_importance_results=compiled_feature_importance)

    def save_results(self, condition_name: str):
        path_to_save = self.path / f"{condition_name}_experiment_results"
        print(f'saving results to {path_to_save}')
        path_to_save.mkdir(parents=True, exist_ok=True)
        for key, df in self.experiment_results.items():
            df.to_parquet(f'{path_to_save}/{key}.parquet')
            print(f'saved {key} in {path_to_save}')


class XGBNormRegExperimenterFactory:
    def __init__(self, path: Path):
        self.path = path
        self.analyzer = XGBNormRegAnalyzer
        self.results = XGBRegrResults

    def create_experimenter(self) -> XGBNormRegExperimenter:
        return XGBNormRegExperimenter(path=self.path, analyzer=self.analyzer, results_reader=self.results)
