import logging
import hydra
from functools import partial
import yaml
from pathlib import Path
import xgboost as xgb


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# local imports

from src.data_processing.preprocessing.pandas_preprocessors import *
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe

from omegaconf import DictConfig, OmegaConf
def df_pipeline(df, query=None):
    '''pandas preprocessing specific to this experiment'''
    drop_columns = ["action", "sex", "trial_count", "trial"]
    df_ = (
        df
        .query(query)
        .pipe(calculate_max_min_signal)
        .pipe(calculate_percent_avoid)
        .drop(columns=drop_columns)
        .pipe(expand_df)
        .drop(columns=['mouse_id', 'day'])
    )
    return df_
# ids = assign_ids(df)


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
    params = {'experiment_name': experiment_name,
              'best_params': best_params}
    with open(experiment_path / 'params.yaml', 'w') as file:
        yaml.dump(params, file)
    logging.info(f'Parameters written to {experiment_path}/params.yaml')
    logging.info(f'Writing results to {experiment_path}/hyper_opt_results.parquet')
    pd.to_parquet(results, experiment_path / 'hyper_opt_results.parquet')
    return None

@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    EXPERIMENT_NAME = cfg.experiment_config.experiment_name

    DATA_PATH = Path(cfg.quest_config.data_path)
    MAIN_DIR = Path(cfg.quest_config.main_dir)
    EXPERIMENT_PATH = Path(cfg.quest_config.experiment_dir)

    logging.info(f"Experiment name: {EXPERIMENT_NAME}")
    
    queried_df_pipeline = partial(df_pipeline, query=cfg.experiment_config.experiment_query)
    

    PROCESSOR_PIPE = (ClassifierPipe(DATA_PATH)
                      .read_raw_data()
                      .pandas_pipe(queried_df_pipeline)
                      .split_by_ratio(target='ratio_avoid')
                      .transform_data()
                      )
    SEARCH_SPACE = {
        "n_estimators": hp.choice('n_estimators', [50, 100, 150, 200, 250]),
        "learning_rate": hp.choice('learning_rate', np.arange(0.005, 1.0, 0.5)),
        "max_depth": hp.choice('max_depth', np.arange(3, 15, 3)),
        "min_child_weight": hp.choice('min_child_weight', np.arange(1, 10, 1)),
        "gamma": hp.choice('gamma', np.arange(0, 5, 1)),
        # "booster": hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        "subsample": hp.choice('subsample', np.arange(0, 1, 0.2)),
        "reg_lambda": hp.choice('reg_lambda', np.arange(0, 5, 0.2))
    }
    best_params, results = hyperopt_experiment(processor=PROCESSOR_PIPE,
                        space=SEARCH_SPACE,
                        max_evals=1000)
    
    save_results(best_params=best_params,
                 results=results,
                 experiment_name=EXPERIMENT_NAME,
                 experiment_path=EXPERIMENT_PATH)

if __name__ == "__main__":
    main()
