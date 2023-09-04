
import os
import pandas as pd
import logging

def create_dir(path):
    """# Summary
    creates a directory if it does not exist

    ### Args:
        path (_type_): _description_
    """
    if path is not None and not os.path.exists(path):
        os.makedirs(path)

    return path


def create_new_directory(directory_extension, root_dir: str,) -> str:
    """
    # Summary
    creates new empty directory for file management downstream
    ## Args
    directory extensions:  directory names to be created
    root_dir:str - path for new directory to be stored

    ## Returns: None
    if directory does not already exists, creates new directory and prints directory created

    if directory already exists, does nothing and prints "directory already exists"
    ### example
        >>> create_new_dir( "my_folder_1", root_dir = "my/root/directory")
        >>> "my/root/directory/my_folder_1"


    """

    new_directory = os.path.join(root_dir, directory_extension)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    return new_directory


def create_directories(*paths):
    """creates new directories in batch using create_new_directory function"""
    for path in paths:
        create_new_directory(path)


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

        if isinstance(df, pd.Series):
            df.to_frame().to_parquet(path, compression='gzip')
        else:
            df.to_parquet(path, compression='gzip')
            
def set_up_directories(*dirs):
        """
    Create directories if they do not exist.

    Parameters
    ----------
    *dirs : str
        The names of the directories to create.

    Returns
    -------
    None

    Notes
    -----
    This function creates directories with the specified names if they do not already exist. If a directory already exists with the same name, no action is taken. The `exist_ok` parameter is set to `True` to prevent an error from being raised if the directory already exists.

    Examples
    --------
    >>> set_up_directories('my_dir/raw_data', 'my_dir/logs')
    """
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def set_up_logger(file_path):
        """
    Set up a logger that writes to a file.

    Parameters
    ----------
    file_path : str
        The path to the log file.

    Returns
    -------
    None

    Notes
    -----
    This function sets up a logger that writes log messages to a file at the specified file path. The logger is configured to write messages at the DEBUG level and uses the following format:

    [%(asctime)s] %(levelname)s - %(message)s

    The log file is created if it does not exist, and is truncated if it already exists.

    Examples
    --------
    >>> set_up_logger('my_log.txt')
    """

    logging.basicConfig(filename=file_path,
                    filemode='w',
                    level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s - %(message)s')