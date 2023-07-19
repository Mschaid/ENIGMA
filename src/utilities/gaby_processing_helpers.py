import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def merge_sex_data(data, path):
    def tweak_sex_data(df):
        return (
            df
            .rename(columns=lambda c: c.replace(' ', '_').lower())
            .assign(mouse_id=lambda df: df['mouse_id'].str.replace("-", "_").astype('category')
                    )
        )

    sex_data = pd.read_excel(path)
    sex_data = tweak_sex_data(sex_data)

    return data.merge(sex_data, on='mouse_id', how='left')


def merge_latency_data(data, path):
    def tweak_lat_data(df):
        return (df.
                rename(columns=lambda c: c.replace(' ', '_').lower())
                .rename(columns={'mouse_': 'mouse_id'})
                .assign(mouse_id=lambda df_: df_['mouse_id'].str.replace("-", "_").astype('category'),
                        event=lambda df_: df_['event'].str.lower())
                )
    latency_data = pd.read_excel(path)
    latency_data = tweak_lat_data(latency_data)
    return (data
            .merge(latency_data, on=['mouse_id', 'day', 'trial', 'event'], how='left')
            .assign(latency=lambda df: df['latency'].fillna(0))
            )


def full_data_feature_extraction(df):
    return (
        df.assign(
            learning_phase=lambda df_: pd.cut(df_.trial,
                                              bins=[-1, 10, 20, 31],
                                              labels=["early", "mid", "late"])
            .astype("category").cat.codes,
            mouse_id=lambda df_: LabelEncoder().fit_transform(df_.mouse_id)
        ))


def assign_cumulative_trials(df):

    # calculate total trials per mouse and store in dict to call later
    total_trials = {mouse: df.groupby(by=['mouse_id', 'day', 'trial']).count(
    ).query("mouse_id == @mouse").shape[0] for mouse in df.mouse_id.unique()}
    # function to generate cumulative trial count for each mouse and store as independent dataframe: used to merge with original dataframe

    def get_trial_count(df, mouse, trials_dict):
        return (
            df
            .query("mouse_id == @mouse")
            .groupby(by=['day', 'trial'], as_index=False).mean()
            .sort_values(by=['day', 'trial'])
            .assign(trial_count=range(0, trials_dict[mouse]))
        )
    # concatenate all mouse dataframes into one and select only relevant columns to me merge with original dataframe
    data_agg_trials = pd.concat([get_trial_count(df, mouse, total_trials) for mouse in df.mouse_id.unique()])[
        ['mouse_id', 'day', 'trial', 'trial_count']]
    data_merged = df.merge(
        data_agg_trials[["mouse_id", "trial", "trial_count", "day"]], how="left")
    return data_merged
