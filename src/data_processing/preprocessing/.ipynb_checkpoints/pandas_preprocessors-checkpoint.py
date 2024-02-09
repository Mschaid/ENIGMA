import numpy as np
import pandas as pd

from typing import List, Callable


def check_for_columns_to_drop(df, cols_to_check):
    cols_to_drop = [c for c in cols_to_check if c in df.columns.to_list()]
    return cols_to_drop


def flatten_dataframe(df):
    """Flatten a multi-indexed dataframe to a single index."""
    df.columns = ['_'.join(col) for col in df.columns.values]
    df = df.reset_index()
    return df


def strip_columns(df, strip_char: str = '_'):
    """
    Strip a character from the end of all column names.
    defaults to _ 
    """
    df.columns = [col.rstrip(strip_char) for col in df.columns]
    return df


def get_features(df, target):
    """ 
    Get a list of features from a dataframe, excluding the target column.
    """
    features = df.columns.tolist()
    features.remove(target)
    return features


def filter_columns_by_search(df: pd.DataFrame, search_term: str) -> List:
    filtered_cols = [col for col in df.columns if search_term in col]
    return filtered_cols


def calculate_max_min_signal(df, cols_to_drop=[]):

    events = filter_columns_by_search(df, 'event')
    actions = filter_columns_by_search(df, 'action')
    mouse = filter_columns_by_search(df, 'mouse')
    sensors = filter_columns_by_search(df, 'sensor')
    sex = filter_columns_by_search(df, 'sex')
    trial_count_bins = filter_columns_by_search(
        df, 'trial_count_bins')
    cols_to_group = mouse+events+actions+sensors+sex + \
        trial_count_bins+['day', 'trial_count', 'trial']

    df = (
        df
        .assign(
            neg_signal=lambda df_: np.where(df_.signal < 0, df_.signal, 0),
            pos_signal=lambda df_: np.where(df_.signal > 0, df_.signal, 0)
        )
        .groupby(by=cols_to_group, as_index=False)
        .agg(
            {
                "signal": ["max", "min", np.trapz],
                "pos_signal": np.trapz,
                "neg_signal": np.trapz
            }
        )
        .pipe(flatten_dataframe)
        .rename(columns=lambda c: c.strip("_"))
        .drop(columns='index')
    )
    df = df.drop(columns=cols_to_drop)
    return df


def calculate_percent_avoid(df):

    cols_to_check = ['sex', 'sensor', 'trial_count']
    cols_to_drop = check_for_columns_to_drop(df, cols_to_check)

    new_df = (df
              .drop(columns=cols_to_drop)
              .replace({"action": {"avoid": 1, "escape": 0}})
              .groupby(by=["mouse_id", "day", "event"], as_index=False).mean(numeric_only = True)
              .rename(columns={"action": "ratio_avoid"})
              .drop_duplicates(subset=["mouse_id", "day"], keep="last")[["mouse_id", "day", "ratio_avoid"]]
              .reset_index(drop=True)
              )

    merged_df = df.merge(new_df, on=["mouse_id", "day"], how="left")
    return merged_df


def debug_df(df):
    print(df)
    return df


def expand_df(df):

    pivot_df = (
        df.pivot_table(index=["ratio_avoid", "day"],
                       columns=["sensor", "event"],
                       values=[col for col in df.columns if col not in ["ratio_avoid", "sensor"]])
        # .drop(columns = ["mouse_id", "day"])

    )
    pivot_df.columns = ["_".join(col) for col in pivot_df.columns]
    return pivot_df.reset_index(drop = False)


def max_trials(df):
    max_trials = (df[["mouse_id", "day", "trial", "action"]]
                  .query("action == 'avoid'")
                  .groupby(by=["mouse_id", "day", "action"])
                  .max()
                  .reset_index()
                  .drop(columns=["action"])
                  .rename(columns={"trial": "max_trial"})

                  )
    max_trials
    return (
        df
        .merge(max_trials, on=["mouse_id", "day"], how="left")
        .drop_duplicates(
            subset=["mouse_id", "day"],
            keep="last")
        .reset_index()
    )


def xgb_reg_signal_params_only_pd_preprocessor(df: pd.DataFrame, query: str = None, cls_to_drop: List = None) -> pd.DataFrame:
    '''pandas preprocessing specific to this experiment'''

    if cls_to_drop is None:
        cls_to_drop = ['mouse_id']
    drop_columns = ["action", "sex", "trial_count", "trial"]
    df_ = (
        df
        .query(query)
        .pipe(calculate_max_min_signal)
        .pipe(calculate_percent_avoid)
        .drop(columns=drop_columns)
        .drop(columns=cls_to_drop)
        .pipe(expand_df)
    
    )
    return df_


def normalize_by_baseline(df, baseline_window="time>-5 & time<-2", time_query="time>=-0 & time <= 10"):
    group_by_columns = [col for col in df.columns if col not in ["signal"]]
    normalized_values = (
        df
        .query(baseline_window)
        .groupby(by=group_by_columns, as_index=False)
        .mean(numeric_only = True)
        .assign(mean_signal=lambda df_: df_.signal)
        .drop(columns=['time', 'signal', 'sex', 'latency', 'learning_phase'])
    )

    normalzied_data = (pd
                       .merge(df, normalized_values, on=[
                           'mouse_id', 'day', 'event', 'sensor', 'trial_count',], how='left')
                       .assign(signal=lambda df_: df_.signal - df_.mean_signal,
                               trial=lambda df_: df_.trial_x,
                               action=lambda df_: df_.action_x)
                       .drop(columns=["mean_signal", "trial_x", "trial_y", "action_x", "action_y"])
                       .query(time_query)
                       )
    return normalzied_data
    # return group_by_columns
def print_cols(df):
    print(df.columns)
    return df


    return df_
def normalized_preprocessor(data: pd.DataFrame, normalizer: Callable, query: str = None, experiment_cols_to_drop: List = None)-> pd.DataFrame:
    if not experiment_cols_to_drop:
        experiment_cols_to_drop = []
    constants_to_drop =  ["action", "trial", "trial", "sex", "mouse_id",  "trial_count"]
    
    processed_data = (
        data
        .query(query)
        .pipe(normalizer)
        .pipe(calculate_max_min_signal)
        .pipe(calculate_percent_avoid)
        .assign(event = lambda df_: df_.event.replace({"avoid": "cross", "escape": "cross"}))
        .drop(columns = constants_to_drop)
        .pipe(expand_df)
        .drop(columns = experiment_cols_to_drop)
    )
    return processed_data