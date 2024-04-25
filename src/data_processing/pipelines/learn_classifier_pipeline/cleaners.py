import polars as pl


from pathlib import Path


def clean_freeze_type(x):
    if x == 'ITI':
        return 'iti'
    elif x == "during_cue":
        return 'cue'
    else:
        return x


def clean_mouse_numb_cols(x):
    try:
        return int(x)
    except:
        return int(x.strip('d'))


def clean_freeze_by_trial_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .select(['Cage', 'Subject', 'Day', 'freeze_type', 'epoch_outcome', 'trial_freeze_duration'])
        .rename({"Cage": 'cage',
                 'Subject': 'mouse_numb',
                 'Day': 'day'})
        .with_columns(
            pl.col('cage').cast(pl.Int64),
            pl.col('mouse_numb').apply(clean_mouse_numb_cols),
            pl.col('day').cast(pl.Int64),
            pl.col('freeze_type').apply(
                clean_freeze_type).cast(pl.Categorical),
            pl.col('epoch_outcome').cast(pl.Categorical),
        )
    )


def clean_freeze_by_day_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .select(['Cage', 'Subject', 'Day', 'freeze_type', 'epoch_outcome', 'total_freeze_duration', 'total_perc_frozen', 'total_perc_frozen_shock_limited'])
        .rename({"Cage": 'cage',
                 'Subject': 'mouse_numb',
                 'Day': 'day'})
        .with_columns(
            pl.col('cage').cast(pl.Int64),
            pl.col('mouse_numb').apply(clean_mouse_numb_cols),
            pl.col('day').cast(pl.Int64),
            pl.col('freeze_type').apply(
                clean_freeze_type).cast(pl.Categorical),
            pl.col('epoch_outcome').cast(pl.Categorical),
        )
    )

# def clean_freeze_by_trial_data


def clean_cue_da_fp_data(df):
    return (
        df
        .with_columns(
            [
                pl.col('mouse_id')
                .str.split_exact("_", 1)
                .struct.rename_fields(['cage', 'mouse_numb']).alias('fields'),
            ]
        ).unnest("fields")
        .with_columns(
            pl.col("mouse_numb").cast(pl.Int64),
            pl.col('cage').cast(pl.Int64),
            pl.col('event').cast(pl.Categorical),
            pl.col('sensor').cast(pl.Categorical),
            pl.col('day').cast(pl.Int64)
        )
        .drop('mouse_id')
        .filter(
            (pl.col('event') == 'cue')
            &
            (pl.col('sensor') == 'DA')
        )
        .drop_nulls()
    )


def clean_core_subjects(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.rename({'id': 'mouse_numb'})
    )


def clean_percent_avoid_excel(path, save=False, path_to_save=None, return_frame=False):

    perc_avoid = pl.read_excel(path)
    clean_frame = (perc_avoid
                   .rename({'Day': 'day'})
                   .melt(id_vars='day', variable_name='id', value_name='perc_avoid')
                   .with_columns(
                       [
                           pl.col('id')
                           .str.split_exact("-", 1)
                           .struct.rename_fields(['cage', 'mouse_numb']).alias('fields'),
                       ]
                   ).unnest("fields")
                   .with_columns(
                       pl.col('cage').cast(pl.Int64),
                       pl.col('mouse_numb').cast(pl.Int64)
                   )
                   .drop('id')
                   .select(['day', 'cage', 'mouse_numb', 'perc_avoid'])
                   )
    if save:
        clean_frame.write_parquet(path_to_save/'percent_avoid.parquet')
    if return_frame:
        return clean_frame
