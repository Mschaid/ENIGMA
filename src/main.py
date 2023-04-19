import logging
import os

from src.processors.SchemaBuilder import SchemaBuilder


def set_logger_config(directory, file_name,  format: str = '[%(asctime)s] %(levelname)s-%(message)s', level=logging.INFO) -> None:

    logging_directory = os.path.join(directory, file_name)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_directory,
        level=level,
        format=format
    )
    return logger


main_logger = set_logger_config(
    directory='/Users/michaelschaid/GitHub/dopamine_modeling/results/logs',
    file_name='test.log')

if __name__ == '__main__':
    main_logger.info(f"test log message")
    sb = SchemaBuilder(logger=main_logger)
    sb.request_directory()
