# Imports

# > Standard Library
import logging


class TensorFlowLogFilter(logging.Filter):
    """
    Filter to exclude specific verbose TensorFlow messages from logs.

    This helps keep logs clean by ignoring known, non-critical spammy messages
    that frequently appear during model training or inference.

    Inherits from
    --------------
    logging.Filter
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine whether a log record should be logged.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to evaluate.

        Returns
        -------
        bool
            True if the message should be logged, False otherwise.
        """
        exclude_phrases = [
            "Reduce to /job:localhost/replica:0/task:0/device:CPU:",
            "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence",
        ]
        return not any(phrase in record.msg for phrase in exclude_phrases)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with a consistent format and level.

    Also sets up a filter to suppress known noisy TensorFlow log messages.

    Parameters
    ----------
    level : str, optional
        Logging level as a string (e.g., "DEBUG", "INFO"). Defaults to "INFO".

    Returns
    -------
    logging.Logger
        A logger instance configured for general application use.
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

    # Remove noisy TensorFlow messages
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger("app")
