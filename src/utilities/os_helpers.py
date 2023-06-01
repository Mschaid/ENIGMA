
import os
import pandas as pd



def create_dir(path):
    """# Summary
    creates a directory if it does not exist

    ### Args:
        path (_type_): _description_
    """
    if path is not None and not os.path.exists(path):
        os.makedirs(path)
        
    return path
def create_new_directoy(file_path: str, new_dir_extension=None) -> str:
    """
    # Summary
    creates new empty directory for file management downstream
    ## Args
    file_path:str - path for new directory to be stored
    new_dir__ext:str - name of new created directory
    ## Returns: None
    if directory does not already exists, creates new directory and prints directory created

    if directory already exists, does nothing and prints "directory already exists"
    ### example
        >>> create_new_dir(file_path='/Users/user/Desktop/', new_dir_ext='my_folder')
        >>>'/Users/user/Desktop/my_folder'
        >>> 'directroy created'
    """
    new_directory = os.path.join(file_path, new_dir_extension)
    print(new_directory)

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    return new_directory

def save_dataframes_to_parquet(*dataframes, path_to_save):
    """

    Save the provided DataFrames to Parquet files.

    Parameters
    ----------
        *dataframes : Tuple[Tuple[str, pd.DataFrame]]
        DataFrames to be saved, along with their corresponding variable names.
        Each DataFrame should be provided as a tuple containing the variable name as a string and the DataFrame itself.
        path_to_save : str
        The path to save the Parquet files.
    Examples
    --------
    >>> save_dataframes_to_parquet(    
        ('X_train', X_train),
        ('X_test', X_test),
        ('y_train', y_train),
        ('y_test', y_test)
        ,path_to_save='/projects/p31961/dopamine_modeling/data/prototype_data')


    """

    for df_name, df in dataframes:
        path = os.path.join(path_to_save, f'{df_name}.parquet.gzip')

        if isinstance(df,pd.Series):
            df.to_frame().to_parquet(path, compression='gzip')
        else:
            df.to_parquet(path, compression='gzip')
        

        
