import os
import numpy as np
import pandas as pd
import pyarrow as pa

from src.data_processing.processors.SchemaBuilder import SchemaBuilder
from src.data_processing.processors.GabyProcessor import GabyProcessor

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
    aggregated_data = aggregated_data.reset_index(drop=True)
    
    save_path = os.path.join(path, 'downsampled_aggregated_data.parquet.gzp')
    aggregated_data.to_parquet(save_path, compression='gzip')


if __name__ == '__main__':
    main()
