
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
def create_new_directoy(*directory_extensions, root_dir: str,) -> str:
    """
    # Summary
    creates new empty directory for file management downstream
    ## Args
    directory extensions: *args - list of directory names to be created
    root_dir:str - path for new directory to be stored

    ## Returns: None
    if directory does not already exists, creates new directory and prints directory created

    if directory already exists, does nothing and prints "directory already exists"
    ### example
        >>> create_new_dir( "my_folder_1", "my_folder_2", root_dir = "my/root/directory")
        >>> "my/root/directory/my_folder_1"
        >>> "my/root/directory/my_folder_2"


    """
    
    for extension in directory_extensions:
        new_directory = os.path.join(root_dir, extension)
        if not os.path.exists(new_directory):
                os.makedirs(new_directory)

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
        

        
