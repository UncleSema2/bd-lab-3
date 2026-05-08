import logging
import os
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = os.path.join(os.getcwd(), "logfile.log")


class Logger:

    def __init__(self, show: bool = True) -> None:
        self.show = show

    def get_console_handler(self) -> logging.StreamHandler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        return console_handler

    def get_file_handler(self) -> logging.FileHandler:
        file_handler = logging.FileHandler(LOG_FILE, mode="w")
        file_handler.setFormatter(FORMATTER)
        return file_handler

    def get_logger(self, logger_name: str):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        if self.show:
            logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        logger.propagate = False
        return logger
