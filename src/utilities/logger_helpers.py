import logging
import os

def set_logger_config(directory, file_name,  format: str = '[%(asctime)s] %(levelname)s-%(message)s', level=logging.INFO) -> None:

    logging_directory = os.path.join(directory, file_name)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_directory,
        level=level,
        format=format
    )
    return logger
