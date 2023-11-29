import logging
import hydra
import os
from pathlib import Path
import xgboost as xgb


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
# local imports

from src.data_processing.preprocessing.pandas_preprocessors import *
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe
from omegaconf import DictConfig, OmegaConf


def df_pipeline(df):
    '''pandas preprocessing specific to this experiment'''
    drop_columns = ["action", "sex", "trial_count", "trial"]
    df_ = (
        df
        .query("sensor=='DA' & sensor == 'D1'")
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
                                  processor.y_dev, cv=5)
        mean_score = np.mean(scores)
        return {'loss': mean_score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    logging.info('Hyperopt complete')
    logging.info(f'Best params: {space_eval(space, best)}')

    # best_trials_path = os.path.


@hydra.main(version_base=None,
            config_path="conf",
            config_name="quest_config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    EXPERIMENT_NAME = cfg.quest_config.experiment_name

    DATA_PATH = Path(cfg.quest_config.data_path)
    MAIN_DIR = Path(cfg.quest_config.main_dir)
    EXPERIMENT_PATH = Path(cfg.quest_config.experiment_dir)

    logging.info(f"Experiment name: {EXPERIMENT_NAME}")

    PROCESSOR_PIPE = (ClassifierPipe(DATA_PATH)
                      .read_raw_data()
                      .pandas_pipe(df_pipeline)
                      .split_by_ratio(target='ratio_avoid')
                      .transform_data()
                      )
    SEARCH_SPACE = {
        "n_estimators": hp.choice('n_estimators', [50, 100, 150, 200, 250]),
        "learning_rate": hp.choice('learning_rate', np.arange(0.05, 0.2, 0.5)),
        "max_depth": hp.choice('max_depth', np.arange(3, 15, 3)),
        "min_child_weight": hp.choice('min_child_weight', np.arange(1, 10, 1)),
        "gamma": hp.choice('gamma', np.arange(0, 5, 1)),
        "booster": hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        "subsample": hp.choice('subsample', np.arange(0, 1, 0.2)),
        "reg_lambda": hp.choice('reg_lambda', np.arange(0, 5, 0.5))
    }
    hyperopt_experiment(processor=PROCESSOR_PIPE,
                        space=SEARCH_SPACE,
                        max_evals=1000)


if __name__ == "__main__":
    main()
