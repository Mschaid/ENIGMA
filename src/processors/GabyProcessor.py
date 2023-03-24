import os
import numpy as np
import pandas as pd


from src.SchemaBuilder import SchemaBuilder


class GabyProcessor:
    def __init__(self, path):
        self.path = path

    def get_meta_data(self):
        split_path = os.path.basename(self.path).split("_")
        self.meta_data = {'mouse_id': split_path[0].replace("-", "_"),
                          'day': int(split_path[1].replace("Day", "")),
                          'event': split_path[2].lower(),
                          'sensor': split_path[3]
                          }
        return self

    def get_data(self):
        df = pd.read_hdf(self.path)

        def rename_trial_columns(df):
            df.columns = [f"trial_{i}" for i in range(len(df.columns))]
            return df

        self.data = (
            df
            .drop(columns=['timestamps', 'mean', 'err'])
            .pipe(rename_trial_columns)
            .assign(
                time=lambda df_: np.linspace(-25, 20, df_.shape[0]),
                **self.meta_data)
            .melt(id_vars=['mouse_id', 'day', 'event', 'sensor', 'time'], var_name='trial', value_name='signal')

        )
        return self


def process_data(file):
    gp = GabyProcessor(file)
    gp.get_meta_data()
    gp.get_data()
    return gp.data


def main():
    path = input("Enter path to data: ")
    builder = SchemaBuilder(path)
    builder.collect_data_files(search_keys=['D1, D2, DA'], file_types=['h5'])

    print('reading data')
    data = [process_data(file) for file in builder.matched_file_types]
    print('aggregate data')
    aggregated_data = pd.concat(data)
    save_path = os.path.join(path, 'aggregated_data.h5')
    aggregated_data.to_hdf(save_path, key='data',
                           mode='w', complib='zlib', complevel=9)
    print(aggregated_data)


if __name__ == '__main__':
    main()
