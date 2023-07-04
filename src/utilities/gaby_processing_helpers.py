import pandas as pd
import numpy as np


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
    latency_data = pd.read_xcel(path)
    latency_data = tweak_lat_data(latency_data)
    return (data
            .dropna()
            .merge(latency_data, on=['mouse_id', 'day', 'trial', 'event'], how='left')
            .assign(latency=lambda df: df['latency'].fillna(0))
            )
