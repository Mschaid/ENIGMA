
import multiprocessing as mp
from pathlib import Path
from abc import ABC, abstractmethod


from typing import Dict, List, NewType
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegAnalyzer import XGBRegAnalyzer, XGBRegAnalyzerFactory
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegrResults import XGBRegrResults
from src.data_processing.model_analyzers.experimenters.experimenters import XGBRegExperimenter, XGBRegExperimenterFactory

ExperimentConditions = NewType('ExperimentCondition', Dict[str, List[str]])


class BatchExperimeter(ABC):
    @abstractmethod
    def __init__(self, main_path):
        pass

    @abstractmethod
    def get_experiment_directories(self):
        pass

    @abstractmethod
    def set_up_experimenters(self):
        pass

    @abstractmethod
    def run_experiments(self):
        pass


def custom_multi_processor(func):
    def wrapper(map_to):
        with mp.Pool(mp.cpu_count()) as p:
            p.map(func, map_to)
            p.close()
            p.join()
    return wrapper


class XGBRegBatchExperimenter(BatchExperimeter):
    def __init__(self, main_path, experiment_conditions: ExperimentConditions):
        self.main_path = Path(main_path)
        self.experiment_conditions = experiment_conditions
        self.xgb_regr_analyzer_factory = XGBRegExperimenterFactory

    def get_experiment_directories(self):
        experiment_directoires = [d for d in self.main_path.rglob(
            '*') if d.is_dir() and not d.name.startswith('.')]
        return experiment_directoires

    # @custom_multi_processor
    def set_up_experimenters(self, directories: List[Path]):
        experimenters = [self.xgb_regr_analyzer_factory(
            d).create_experimenter()for d in directories]
        return experimenters

    # @custom_multi_processor
    def run_experiments(self, experimenters: List[XGBRegExperimenter], number_of_runs: int):
        for exp in experimenters:
            for condition_name, cls_to_drop in self.experiment_conditions.experiment_conditions.items():
                exp.run_experiment(number_of_runs, cls_to_drop)
                exp.save_results(condition_name)
