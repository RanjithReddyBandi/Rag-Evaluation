"""Logging configuration for the Ragas validation tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(
    level_name: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure application logging and return the named package logger.

    Parameters:
        level_name: Logging level name, such as ``DEBUG`` or ``INFO``.
        log_file: Optional file that receives the same messages as the console.

    Returns:
        The configured logger for the Ragas validation package.

    Raises:
        ValueError: If level_name is not a recognized logging level.
        OSError: If the log directory cannot be created or the file cannot be opened.
    """
    normalized_level_name = level_name.strip().upper()
    numeric_level = getattr(logging, normalized_level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level_name}")

    logger = logging.getLogger("ragas_validation")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Avoid adding duplicate handlers when configure_logging is called repeatedly.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug(
        "Logging configured",
        extra={"level": normalized_level_name, "log_file": str(log_file or "")},
    )
    return logger
