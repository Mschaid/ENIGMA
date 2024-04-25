from scipy import signal
import polars as pl
import numpy as np


def compute_dip_duration(df) -> float:
    cue_time_filter = df.filter(
        (pl.col('time') > 0)
        &
        (pl.col('time') < 5)
    )
    time_arr = cue_time_filter.select(pl.col('time')).to_numpy().squeeze()
    sig_arr = cue_time_filter.select(pl.col('signal')).to_numpy().squeeze()
    # find signle min peak and propertiess
    peak, properties = signal.find_peaks(
        -sig_arr, distance=sig_arr.shape[0], width=time_arr, rel_height=1)

    # covert indx to time
    time_convert = 5/time_arr.shape[0]
    peak_duration = properties['widths'] * time_convert
    try:
        assert peak_duration.size != 0
        return peak_duration[0]
    except:
        return 0.0


def compute_avg_dip_duration_for_cue_df(df: pl.DataFrame, save=False, path_to_save=None) -> pl.DataFrame:
    cat_cols = [c for c in df.columns if c not in [
        'signal', 'time', 'event', 'sensor', 'cage']]
    unique_rows = df.select(cat_cols).group_by(
        cat_cols).n_unique().sort('day', 'trial', 'mouse_numb')

    dip_frame_schema = dict(df.schema)
    dip_frame_schema['dip_duration'] = pl.Float64
    dip_frame = pl.DataFrame(schema=dip_frame_schema)
    for r in unique_rows.iter_rows():
        uniq_frame = df.filter(
            (pl.col('day') == r[0])
            &
            (pl.col('trial') == r[1])
            &
            (pl.col('mouse_numb') == r[2])
        )
        dip_dur = compute_dip_duration(uniq_frame)
        uniq_dip_frame = uniq_frame.with_columns(
            pl.lit(dip_dur).alias('dip_duration'))
        dip_frame = pl.concat([dip_frame, uniq_dip_frame], how='vertical')
    return_frame = (
        dip_frame
        .drop('time', 'signal', 'trial')
        .group_by(['day', 'event', 'sensor', 'mouse_numb', 'cage'])
        .agg(pl.mean('dip_duration'))
        .sort('day', 'mouse_numb', 'cage')
    )

    if save:
        return_frame.write_parquet(
            path_to_save / "day_average_dip_duration.parquet")
    return return_frame
