
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


from typing import Callable, List, Optional, Tuple, Dict, Any, Literal, Protocol
from pathlib import Path


# local imports
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegAnalyzer import XGBRegAnalyzer
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

    @property
    def results(self):
        if not self._results:
            self._results = XGBRegrResults(self.path)
        return self._results

    def run_experiment(self):
        pass

    def save_results(self):
        pass


class XGBRegExperimenter:
    def __init__(self, path, analyzer, results):
        self.path = Path(path)
        self.results = results(path)
        self.analyzer = analyzer
        self._experiment_metric_results = None
        self.experiment_results: Dict[str, pd.DataFrame] = {}
        self.analyzer_runs: List[XGBRegAnalyzer] = []

    def run_experiment(self, number_of_runs: 10, cls_to_drop: List[str] = []):
        metric_runs = []
        feature_importance_runs = []
        for run_numb in range(number_of_runs):
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

    def save_results(self):
        path_to_save = self.path / "experiment_results"
        path_to_save.mkdir(parents=True, exist_ok=True)
        for key, df in self.experiment_results.items():

            df.to_parquet(f'{path_to_save}/{key}.parquet')


class ElasticNetRegulairzationHyperopt:
    def __init__(self, analyzer, results):
        self.results = results


# load conifgurations, including best parameters
# run hyperopt on the same model with the best parameters, and search for best L1 and L2 combination
