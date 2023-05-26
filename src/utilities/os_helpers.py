
import inspect
import os
import re



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

    if os.path.exists(new_directory):
        pass
    else:
        os.mkdir(new_directory)



def get_variable_name(variable):
    """
    Retrieve the string representation of a variable name in a Jupyter Notebook.

    Parameters
    ----------
    variable : object
        The variable for which to retrieve the name.

    Returns
    -------
    str or None
        The string representation of the variable name if found, None otherwise.

    Notes
    -----
    - This function is specifically designed for use in a Jupyter Notebook environment.
    - It retrieves the variable name by comparing values in the local variables of the calling frame.
    - If the variable is not found in the local variables, None is returned.
    - This approach relies on comparing variable values and assumes that there are no two variables with the same value.
    - It also assumes that the variable is defined within the current scope or an outer scope.

    Example
    -------
    >>> my_variable = 42
    >>> variable_name = get_variable_name(my_variable)
    >>> print(variable_name)
    'my_variable'
    """
    frame = inspect.currentframe().f_back
    variables = frame.f_locals
    
    for name in variables:
        value = variables[name]
        if value is variable:
            return name
    
    return None

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
        

        
