"""Logging configuration and utilities."""
import logging
import sys
from pathlib import Path

_logger_initialized = False

def setup_logger(
    name: str = "antipompeii",
    level: str = "INFO",
    log_file: str = "./logs/app.log",
    console_output: bool = True
) -> logging.Logger:
    """Set up application logger."""
    global _logger_initialized

    logger = logging.getLogger(name)

    if _logger_initialized:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # File handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _logger_initialized = True
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"antipompeii.{name}")


def get_module_logger(name, logger: logging.Logger | None = None) -> logging.Logger:
    """Return *logger* if provided, otherwise create a module-level logger.

    Accepts two calling conventions used across the codebase:
      get_module_logger("module_name")
      get_module_logger("module_name", existing_logger)
      get_module_logger(existing_logger)   # legacy: logger passed as name
    """
    if isinstance(name, logging.Logger):
        return name
    if logger is not None:
        return logger
    log = logging.getLogger(name if isinstance(name, str) else "antipompeii")
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log
