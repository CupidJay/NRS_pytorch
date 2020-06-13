import logging
import os
import sys
import colorlog

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch_formatter = colorlog.ColoredFormatter("%(asctime)s %(name)s %(levelname)s: %(log_color)s %(message)s", log_colors=log_colors_config)
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
