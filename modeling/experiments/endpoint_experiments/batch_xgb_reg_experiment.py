from src.data_processing.model_analyzers.experimenters.batch_experimenters import XGBRegBatchExperimenter


def main():

    NUMBER_OF_RUNS = 20
    MAIN_PATH = "/projects/p31961/ENIGMA/results/experiments/endpoint_experiments"
    EXPERIMENMT_CONDITIONS = {
        "with_day": ["mouse_id", "day"],
        "with_out_day": ["mouse_id"]
    }

    bacth_experimenter = XGBRegBatchExperimenter(
        MAIN_PATH, EXPERIMENMT_CONDITIONS)

    experiment_directories = bacth_experimenter.get_experiment_directories()
    experimenters = bacth_experimenter.set_up_experiments(
        experiment_directories)
    bacth_experimenter.run_experiments(experimenters, NUMBER_OF_RUNS)


if __name__ == '__main__':
    main()
