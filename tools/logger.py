"""
Code inspired from https://techies-world.com/how-to-redirect-stdout-and-stderr-to-a-logger-in-python/
"""

import logging
import sys
import os


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def config_logger(dir_path):
    """
    Configure logger.
    Redirect stdout and stderr to the logger using StreamToLogger class.

    Args:
        dir_path: string, path to directory where logger output will be stored.
    """

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=os.path.join(dir_path, 'log.out'),
        filemode='w'
    )

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    return stdout_logger
