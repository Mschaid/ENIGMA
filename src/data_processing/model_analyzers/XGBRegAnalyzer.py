# Standard library imports
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import xgboost as xgb
import yaml

from src.data_processing.preprocessing.pandas_preprocessors import xgb_reg_signal_params_only_pd_preprocessor
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe


@dataclass
class XGBRegrResults:
    _data_path: str  # ! TODO can generate all of these from the results path

    """
    then analyzer  =  XGBRegAnalyzer(resutls_path)
    -> analyer.results = XGBRegrResults(results_path)
    """
    _results_path: str
    sensor_query: str
    _hyperopt_exper_results: pd.DataFrame = None
    _best_params: Dict[str, Any] = None
    _experiment_name: str = None

    @property
    def data_path(self):
        return Path(self._data_path)

    @property
    def results_path(self):
        return Path(self._results_path)

    def get_params_and_experiment_name(self):
        params = self.results_path / 'params.yaml'
        with open(params, 'r') as f:
            data = yaml.safe_load(f)
            self._best_params = data['best_params']
            self._experiment_name = data['experiment_name']

    @property
    def best_params(self):
        if not self._best_params:
            self.get_params_and_experiment_name()
        return self._best_params

    @property
    def experiment_name(self):
        if not self._experiment_name:
            self.get_params_and_experiment_name()
        return self._experiment_name

    @property
    def hyperopt_exper_results(self):
        if not self._hyperopt_exper_results:
            self._hyperopt_exper_results = pd.read_parquet(
                self.results_path / 'hyper_opt_results.parquet')
        return self._hyperopt_exper_results


class XGBRegAnalyzer:
    def __init__(self, results: XGBRegrResults):
        self.results = results
        self._pipeline = None
        self._xgb_model = None

    @property
    def xgb_model(self):

        if not self._xgb_model:
            self._xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric=['rmse', 'mae'],
                **self.results.best_params)

        return self._xgb_model

    def create_pipeline(self, cls_to_drop: List[str] = None):

        df_processor = partial(xgb_reg_signal_params_only_pd_preprocessor,
                               query=self.results.sensor_query,
                               cls_to_drop=cls_to_drop)
        if not self._pipeline:
            self._pipeline = (ClassifierPipe(self.results.data_path)
                              .read_raw_data()
                              .pandas_pipe(df_processor)
                              .split_by_ratio(target='ratio_avoid')
                              .transform_data()
                              )
        return

    @property
    def pipeline(self):
        if not self._pipeline:
            raise ValueError(
                'Pipeline not created yet, call create_pipeline() first')
        return self._pipeline
