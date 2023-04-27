import io
import logging
import os
import unittest


class LoggingTestRunner(unittest.TextTestRunner):
    def __init__(self, stream=None, descriptions=True, verbosity=1, failfast=False, buffer=False, resultclass=None,
                 warnings=None, *, tb_locals=False):
        super().__init__(stream=io.StringIO(), descriptions=descriptions, verbosity=verbosity, failfast=failfast,
                         buffer=buffer, resultclass=resultclass, warnings=warnings, tb_locals=tb_locals)

    def run(self, test):
        result = super().run(test)
        logger = logging.getLogger(__name__)
        logger.info(self.stream.getvalue())
        return result

def set_logger_config(directory, file_name,  format: str = '[%(asctime)s] %(levelname)s-%(message)s', level=logging.INFO) -> None:

    logging_directory = os.path.join(directory, file_name)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_directory,
        level=level,
        format=format
    )
    return logger
