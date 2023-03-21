
import os


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



