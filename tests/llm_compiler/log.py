import logging
import os
from datetime import datetime


def setup_logging(log_dir: str = ".logs") -> logging.Logger:
    """Configure logging to output to both console and file.

    Args:
        log_dir: Directory to store log files.

    Returns:
        Configured logger instance.
    """
    # Normalize path to ensure relative paths are resolved correctly
    if not os.path.isabs(log_dir):
        # If it starts with '.', use as relative; otherwise convert to absolute
        if log_dir.startswith("."):
            log_dir = os.path.normpath(os.path.join(os.getcwd(), log_dir))
        else:
            log_dir = os.path.join(os.getcwd(), log_dir)

    # Create the log directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)

    # Build a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llm_compiler_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("llm_compiler")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Log record format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized, log file: {log_file}")

    return logger


def get_logger(name: str = "llm_compiler") -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
