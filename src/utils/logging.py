import logging
from logging import FileHandler
from logging import LogRecord
from logging import StreamHandler
import sys


class ConsoleFilter(logging.Filter):
    def filter(self, record: LogRecord) -> bool:
        return record.levelno == logging.INFO or record.levelno == logging.ERROR


def setup_logging():
    file_handler = FileHandler("test.log")
    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter())

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, console_handler],
    )
