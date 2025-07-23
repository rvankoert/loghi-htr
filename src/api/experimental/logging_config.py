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

    This function is idempotent. It can be called multiple times to reset and
    re-apply the logging configuration, which is useful for overriding
    server-specific logging setups (like Hypercorn's).

    It also sets up a filter to suppress known noisy TensorFlow log messages
    and silences verbose third-party loggers.

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

    # Make the function idempotent by removing existing root handlers.
    # This allows us to call it again after a server like Hypercorn has
    # set up its own handlers.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    logging.basicConfig(
        format="[%(processName)s] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=log_level,
    )

    # Suppress verbose messages from third-party libraries.
    # Set them to a level higher than DEBUG to avoid spam.
    logging.getLogger("python_multipart").setLevel(logging.ERROR)
    logging.getLogger("hpack").setLevel(logging.ERROR)

    # Remove noisy TensorFlow messages
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger("app")
