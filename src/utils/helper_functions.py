import logging


def setup_logger(logger_name: str, level: int = logging.INFO) -> logging:
    """
    This method sets up the logger.
    :param logger_name: logger name
    :param level: logger level
    :return: logger: logger object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s::%(name)s::%(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
