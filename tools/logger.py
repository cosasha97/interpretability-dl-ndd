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
    os.makedirs(dir_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=os.path.join(dir_path, 'log.out'),
        filemode='a'
    )

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    return stdout_logger


def str2bool(v):
    """
    Function taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Translate command line input into a boolean.
    :param v: string, command line input
    :return: boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
