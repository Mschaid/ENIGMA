import logging
import hydra
from functools import partial
import yaml
from pathlib import Path
import xgboost as xgb

from src.data_processing.processors.guppy_processors.config_loader import ConfigLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# local imports
from src.data_processing.model_analyzers.experimenters.elastic_net_optimizer import ElasticNetOptimizer

from src.data_processing.preprocessing.pandas_preprocessors import *
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe

from omegaconf import DictConfig, OmegaConf


def hyperopt_experiment(processor, space, max_evals):
    logging.info('Running hyperopt')

    def objective(params):

        model = xgb.XGBRegressor(
            objective='reg:squarederror', eval_metric=['rmse', 'mae'], **params)
        model.fit(processor.X_train, processor.y_train)
        scores = -cross_val_score(model, processor.X_dev,
                                  processor.y_dev, cv=5,
                                  scoring='neg_root_mean_squared_error')
        mean_score = np.mean(scores)

        return {'loss': mean_score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    best_params = space_eval(space, best)
    logging.info('Hyperopt complete')
    logging.info(f'Best params: {space_eval(space, best)}')
    results = pd.DataFrame(trials.results)

    return best_params, results


def save_results(best_params, results,  experiment_name, experiment_path):
    '''Writes experiment parameters to yaml file'''
    for key, value in best_params.items():
        if isinstance(value, np.generic):
            best_params[key] = value.item()

    params = {'experiment_name': experiment_name,
              'best_params': best_params}
    with open(experiment_path / 'params.yaml', 'w') as file:
        yaml.dump(params, file, default_flow_style=False)
    logging.info(f'Parameters written to {experiment_path}/best_params.yaml')
    logging.info(
        f'Writing results to {experiment_path}/hyper_opt_results.parquet')

    results.to_parquet(path=experiment_path / 'hyper_opt_results.parquet',
                       engine='pyarrow', compression='gzip')
    return


@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO: this current overrides everything and it needs to be fixed so it outputs to new directory.
    OmegaConf.to_yaml(cfg)
    EXPERIMENT_NAME = cfg.experiment_name
    ORIG_EXPERIMENT_NAME = cfg.original_experiment_name

    DATA_PATH = Path(cfg.quest_config.data_path)
    MAIN_DIR = Path(cfg.quest_config.main_dir)
    EXPERIMENT_PATH = Path(cfg.quest_config.experiment_dir)
    ORIG_EXPERIMENT_PATH = Path(cfg.quest_config.original_experiment_dir)

    logging.info(f"Experiment name: {EXPERIMENT_NAME}")

    queried_df_pipeline = partial(xgb_reg_signal_params_only_pd_preprocessor, cls_to_drop=[
                                  'day'],  query=str(cfg.experiment_query))

    PROCESSOR_PIPE = (ClassifierPipe(DATA_PATH)
                      .read_raw_data()
                      .pandas_pipe(queried_df_pipeline)
                      .split_by_ratio(target='ratio_avoid')
                      .transform_data()
                      )

    NET_PARAMS = {
        'alpha': hp.uniform('alpha', 0, 5),
        'lambda': hp.uniform('lambda', 0, 5)
    }

    net_optimizer = ElasticNetOptimizer(ORIG_EXPERIMENT_PATH)
    SEARCH_SPACE = net_optimizer.set_experimental_params(
        experimental_params=NET_PARAMS)

    best_params, results = hyperopt_experiment(processor=PROCESSOR_PIPE,
                                               space=SEARCH_SPACE,
                                               max_evals=500)

    save_results(best_params=best_params,
                 results=results,
                 experiment_name=EXPERIMENT_NAME,
                 experiment_path=EXPERIMENT_PATH)
    logging.info('Experiment complete')


if __name__ == "__main__":
    main()
