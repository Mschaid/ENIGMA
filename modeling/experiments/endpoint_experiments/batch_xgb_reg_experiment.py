import multiprocessing as mp
from src.data_processing.model_analyzers.experimenters.batch_experimenters import XGBRegBatchExperimenter
from dataclasses import dataclass
from typing import Dict, List
import logging


@dataclass
class BatchExperimentMetaData:
    main_path: str
    number_of_runs: int
    experiment_conditions: Dict[str, List[str]]
    filter_keys: List[str] = None


def batch_experiment(exp_data: BatchExperimentMetaData):

    batch_experimenter = XGBRegBatchExperimenter(
        exp_data.main_path, exp_data)

    experiment_directories = batch_experimenter.get_experiment_directories(
        exp_data.filter_keys)

    experimenters = batch_experimenter.set_up_experimenters(
        experiment_directories)
    batch_experimenter.run_experiments(experimenters, exp_data.number_of_runs)


def main():

    NUMBER_OF_RUNS = 30
    MAIN_PATH = "/projects/p31961/ENIGMA/results/experiments/endpoint_experiments"
    EXPERIMENMT_CONDITIONS = {
        "with_day": ["mouse_id"]
    }
    FILTER_KEYS = ['no_reg', 'elastic_net']

    experiment_data = BatchExperimentMetaData(
        main_path=MAIN_PATH,
        number_of_runs=NUMBER_OF_RUNS,
        experiment_conditions=EXPERIMENMT_CONDITIONS,
        filter_keys=FILTER_KEYS)

    batch_experiment(experiment_data)


if __name__ == '__main__':
    main()
