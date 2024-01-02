
from src.data_processing.processors.guppy_processors.config_loader import ConfigLoader

import pandas as pd
from pathlib import Path

from abc import ABC, abstractmethod
from typing import List


class AggregationStrategy(ABC):
    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    @abstractmethod
    def make_aggregate_dir(self):
        pass

    @abstractmethod
    def aggreate_processed_results_into_df(self):
        pass

    def save_to_aggregate_dir(self):
        pass


class BehaviorAggregationStrategy(AggregationStrategy):
    def __init__(self, configs: ConfigLoader):
        self.configs = configs
        self.aggregate_dir = self.make_aggregate_dir()

    def make_aggregate_dir(self, name: str = 'aggregated_data'):
        aggregate_dir = self.configs.data_path / name
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        return aggregate_dir

    def aggreate_processed_results_into_df(self, search_name: str = 'processed_data', file_type='.parquet') -> pd.DataFrame:
        found_files = self.configs.data_path.rglob(
            f'*{search_name}{file_type}')
        return pd.concat([pd.read_parquet(f) for f in found_files])

    def save_to_aggregate_dir(self, file_name_to_save: str = 'aggregated_behavior_data', data: pd.DataFrame = None):
        if not data:
            aggregate_data = self.aggreate_processed_results_into_df()
            aggregate_data.to_parquet(
                self.aggregate_dir / f'{file_name_to_save}.parquet')

        if data:
            data.to_parquet(self.aggregate_dir /
                            f'{file_name_to_save}.parquet')


def aggregate_data(configs: ConfigLoader, aggregation_strategy: AggregationStrategy, data=None):
    aggregation_strategy = aggregation_strategy(configs)
    if data:
        aggregation_strategy.save_to_aggregate_dir(data=data)
    else:
        aggregation_strategy.save_to_aggregate_dir()