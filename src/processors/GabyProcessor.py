import os
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

        def rename_trial_columns(df):
            df.columns = [f"trial_{i}" for i in range(len(df.columns))]
            return df

        self.data = (
            pd.read_hdf(self.path)
            .drop(columns=['timestamps', 'mean', 'err'])
            .pipe(rename_trial_columns)
            .melt(var_name='trial', value_name='signal')
            .assign(**self.meta_data)
        )
        return self


def process_data(file):
    gp = GabyProcessor(file)
    gp.get_meta_data()
    gp.get_data()
    return gp.data


def main():
    path = r'R:\Gaby\MetaData_ForMike'
    builder = SchemaBuilder(path)
    builder.collect_data_files(search_keys=['D1, D2, DA'], file_types=['h5'])
    # print(builder.matched_file_types)
    print('reading data')
    data = [process_data(file) for file in builder.matched_file_types]
    print('aggregate data')
    aggregated_data = pd.concat(data)
    print(aggregated_data)


if __name__ == '__main__':
    test_path = r"R:\Gaby\MetaData_ForMike\Day7\142-237_Day7_Avoid_D1_z_score_D1.h5"
    gp = GabyProcessor(test_path)
    gp.get_meta_data()
    gp.get_data()
    print(gp.data)
