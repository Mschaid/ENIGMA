
import multiprocessing as mp
from pathlib import Path
from abc import ABC, abstractmethod


from typing import Dict, List, NewType
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegAnalyzer import XGBRegAnalyzer, XGBRegAnalyzerFactory, XGBNormRegAnalyzer, XGBNormRegAnalyzerFactory
from src.data_processing.model_analyzers.xgb_analyzers.XGBRegrResults import XGBRegrResults
from src.data_processing.model_analyzers.experimenters.experimenters import XGBRegExperimenter, XGBRegExperimenterFactory, XGBNormRegExperimenter, XGBNormRegExperimenterFactory

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


class XGBRegBatchExperimenter(BatchExperimeter):
    def __init__(self, main_path, experiment_conditions: ExperimentConditions, analyzer_factory=XGBRegAnalyzerFactory):
        self.main_path = Path(main_path)
        self.experiment_conditions = experiment_conditions
        self.analyzer_factory = analyzer_factory

    def get_experiment_directories(self, filter_keywords=None):
        if filter_keywords:
            experiment_directoires = [d for d in self.main_path.rglob(
                '*') if d.is_dir() and not d.name.startswith('.') and any(keyword in d.name for keyword in filter_keywords)]
        else:
            experiment_directoires = [d for d in self.main_path.rglob(
                '*') if d.is_dir() and not d.name.startswith('.')]
        return experiment_directoires

    # @custom_multi_processor
    def set_up_experimenters(self, directories: List[Path]):
        experimenters = [self.analyzer_factory(
            d).create_experimenter()for d in directories]
        return experimenters

    # @custom_multi_processor
    def run_experiments(self, experimenters: List[XGBRegExperimenter], number_of_runs: int):
        for exp in experimenters:
            for condition_name, cls_to_drop in self.experiment_conditions.experiment_conditions.items():
                exp.run_experiment(number_of_runs, cls_to_drop)
                exp.save_results(condition_name)


class XGBNormRegBatchExperimenter(BatchExperimeter):
    def __init__(self, main_path, experiment_conditions: ExperimentConditions, experimenter_factory=XGBNormRegExperimenterFactory):
        self.main_path = Path(main_path)

        self.experiment_conditions = experiment_conditions
        self.experimenter_factory = experimenter_factory
    def get_experiment_directories(self, filter_keywords=None):
        if filter_keywords:
            experiment_directoires = [d for d in self.main_path.rglob(
                '*') if d.is_dir() and not d.name.startswith('.') and any(keyword in d.name for keyword in filter_keywords)]
        else:
            experiment_directoires = [d for d in self.main_path.rglob(
                '*') if d.is_dir() and not d.name.startswith('.')]
        return experiment_directoires

    def set_up_experimenters(self, directories: List[Path]):
        experimenters = [self.experimenter_factory(
            d).create_experimenter()for d in directories]
        return experimenters

    def run_experiments(self, experimenters: List[XGBNormRegExperimenter], number_of_runs: int):
        print(
            f"norm conditions: {self.experiment_conditions.experiment_conditions.items()}")
        print("run exp called from norm")
        for exp in experimenters:
            for condition_name, cls_to_drop in self.experiment_conditions.experiment_conditions.items():
                print(condition_name)
                print(cls_to_drop)
                exp.run_experiment(number_of_runs, cls_to_drop)
                exp.save_results(condition_name)
