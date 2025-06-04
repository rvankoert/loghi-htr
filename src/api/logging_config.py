import logging


class TensorFlowLogFilter(logging.Filter):
    """Filter to exclude specific TensorFlow logging messages."""

    def filter(self, record):
        exclude_phrases = [
            "Reduce to /job:localhost/replica:0/task:0/device:CPU:",
            "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence",
        ]
        return not any(phrase in record.msg for phrase in exclude_phrases)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging with the specified level and return a logger instance.
    """
    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = logging_levels.get(level.upper(), logging.INFO)

    logging.basicConfig(
        format="[%(processName)s] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=log_level,
    )

    # Configure TensorFlow logger to use the custom filter
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger("app")  # Return a general app logger
