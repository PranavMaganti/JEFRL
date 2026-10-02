import logging
from logging import FileHandler
from logging import LogRecord
from logging import StreamHandler
import os
from pathlib import Path
import sys


class ConsoleFilter(logging.Filter):
    def filter(self, record: LogRecord) -> bool:
        return record.levelno in (logging.INFO, logging.WARNING, logging.ERROR)


def setup_logging(log_file: Path = Path("jefrl.log")):
    # Per-step messages are logged at DEBUG. Writing them all slows fuzzing down and
    # produces huge logs, so they are only kept when JEFRL_DEBUG=1.
    level = logging.DEBUG if os.environ.get("JEFRL_DEBUG") == "1" else logging.INFO

    file_handler = FileHandler(log_file)
    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter())

    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, console_handler],
    )
