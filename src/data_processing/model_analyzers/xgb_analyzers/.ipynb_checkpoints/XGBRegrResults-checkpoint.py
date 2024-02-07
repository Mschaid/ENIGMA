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
    _results_path: str  # only attribute that is required
    _configs = None
    _best_params = None
    _hyperopt_results = None

    @property
    def results_path(self):
        return Path(self._results_path)

    @property
    def config(self):
        if not self._configs:
            hydra_path = self.results_path / '.hydra'
            configs_path = hydra_path / 'config.yaml'
            with open(configs_path, 'r', encoding='utf-8') as f:
                self._configs = yaml.safe_load(f)
        return self._configs

    def get_params_and_experiment_name(self):
        if not self._best_params:
            params_path = self.results_path / 'params.yaml'
            with open(params_path, 'r') as f:
                data = yaml.safe_load(f)
                self._best_params = data['best_params']
        return self._best_params

    @property
    def best_params(self):
        if not self._best_params:
            self.get_params_and_experiment_name()
        return self._best_params

    @property
    def experiment_name(self):
        return self.config['experiment_name']

    @property
    def experiment_query(self):
        return self.config['experiment_query']

    @property
    def data_path(self):
        return Path(self.config['quest_config']['data_path'])

    @data_path.setter
    def data_path(self, value):
        self.config['quest_config']['data_path'] = str(value)

    @property
    def hyperopt_results(self):
        if not isinstance(self._hyperopt_results, pd.DataFrame):
            path_to_experiment_results = self.results_path / 'hyper_opt_results.parquet'
            self._hyperopt_results = pd.read_parquet(
                path_to_experiment_results)
        return self._hyperopt_results
