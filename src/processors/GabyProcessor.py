import os
import numpy as np
import pandas as pd

from src.processors.SchemaBuilder import SchemaBuilder


class GabyProcessor:
    """ #Summary:
    This class is used to process process and aggregate data from Gaby's fear conditioning experiments

    ## Attributes:
    - path (str): path to the data file
    - meta_data (dict): dictionary containing meta data, this includes mouse_id, experiment day, events, and sensor used
    - data (pd.DataFrame): dataframe containing the aggregated data
    """

    def __init__(self, path):
        ''' #Summary:
        Constructor for the GabyProcessor class

        ## Parameters:
        - path (str): path to the data file
        - meta_data (dict): dictionary containing meta data, this includes mouse_id, experiment day, events, and sensor used
        - data (pd.DataFrame): dataframe containing the aggregated data
        '''
        self.path = path
        self.meta_data = None
        self.data = None

    def get_meta_data(self):
        """
        # Summary:
            reads meta data from path and stores self.meta_data that includes mouse_id, experiment day, events, and sensor used

        ## Returns:
            self

        """

        split_path = os.path.basename(self.path).split("_")
        self.meta_data = {'mouse_id': split_path[0].replace("-", "_"),
                          'day': int(split_path[1].replace("Day", "")),
                          'event': split_path[2].lower(),
                          'sensor': split_path[3]
                          }
        return self

    def get_data(self):
        ''' # Summary:
                reads data from path and stores self.data that includes meta data

        '''
        # reads dataframe from path
        df = pd.read_hdf(self.path)

        def rename_trial_columns(df):
            '''help function that renames trials with prefix trial_'''
            df.columns = [f"trial_{i}" for i in range(len(df.columns))]
            return df

        self.data = (
            df
            # drop columns that are not needed
            .drop(columns=['timestamps', 'mean', 'err'])
            # .iloc[::2, :]
            .pipe(rename_trial_columns)  # rename trials with prefix trial_
            .assign(
                # uniform time colulmn
                time=lambda df_: np.linspace(-25, 20, df_.shape[0]),
                **self.meta_data)  # assign meta data to dataframe
            # reconfigure dataframe
            .melt(id_vars=['mouse_id', 'day', 'event', 'sensor', 'time'], var_name='trial', value_name='signal')
            # this assign is used to reduce memory usage
            .assign(mouse_id=lambda df_: df_['mouse_id'].str.replace("-", "_").astype('category'),
                    day=lambda df_: df_['day'].astype('category'),
                    event=lambda df_: df_['event'].astype('category'),
                    sensor=lambda df_: df_['sensor'].astype('category'),
                    time=lambda df_: df_['time'].astype('float32'),
                    signal=lambda df_: df_['signal'].astype('float32'),
                    trial=lambda df_: df_['trial'].str.replace(
                        "trial_", "").astype('int32')
                    )

        )
        return self


def process_data(file) -> pd.DataFrame:
    """
    # Summary:
    Initiaizes the GabyProcessor class, reads meta data, and processes data 

    ## Returns:
    - data (pd.DataFrame): dataframe containing the aggregated data
    """
    gp = GabyProcessor(file)
    gp.get_meta_data()
    gp.get_data()
    return gp.data


def main():
    # initialize schema builder and collect files to process
    path = input("Enter path to data: ")
    builder = SchemaBuilder(path)
    builder.collect_data_files(
        search_keys=['D1, D2, DA'], file_types=['h5'])

    # collect and process data
    print('reading data')
    data = [process_data(file) for file in builder.matched_file_types]

    # aggregate data and save
    print('aggregate data')
    aggregated_data = pd.concat(data)
    save_path = os.path.join(path, 'aggregated_data.parquet.gzp')
    aggregated_data.to_parquet(save_path, compression='gzip')


if __name__ == '__main__':
    main()
