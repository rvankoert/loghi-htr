# Imports

# > Third-party dependencies
import tensorflow as tf

# > Standard library
import logging
from typing import List, Optional, Tuple, Union


def print_predictions(filename: str,
                      original_text: str,
                      predicted_text: str,
                      normalized_text: Optional[str] = None,
                      char_str: Optional[str] = None) -> None:
    """
    Logs the original and predicted text for a given file, and optionally logs
    Word Beam Search results.

    Parameters
    ----------
    filename : str
        The name of the file for which predictions are made.
    original_text : str
        The original text corresponding to the file.
    predicted_text : str
        The text predicted by the model.
    normalized_text : Optional[str]
        The normalized text, if applicable.
    char_str : Optional[str]
        The result from Word Beam Search, if applicable.

    Notes
    -----
    This function uses logging to display the information. If `char_str` is
    provided, it additionally logs the results from Word Beam Search.
    """

    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info(f"File: {filename}")
    logging.info("")
    logging.info(f"Original text  - {original_text}")
    if normalized_text:
        logging.info(f"Normalized text - {normalized_text}")
    logging.info(f"Predicted text - {predicted_text}")
    if char_str:
        logging.info(f"WordBeamSearch - {char_str}")
    logging.info("")


def print_cer_stats(distances: Tuple[int, int, int],
                    lengths: Tuple[int, int],
                    prefix: str = "") -> None:
    """
    Logs Character Error Rate (CER) statistics including standard, lower case,
    and simplified CER.

    Parameters
    ----------
    distances : Tuple[int, int, int]
        A tuple containing the edit distances for standard, lower case, and
        simplified text.
    lengths : Tuple[int, int]
        A tuple containing the lengths of the original texts for standard and
        simplified text.
    prefix : str, optional
        An optional prefix to add before the CER statistics in the log.

    Notes
    -----
    This function calculates CERs based on provided edit distances and lengths
    and then logs the results using the specified prefix, if any.
    """

    prefix = f"{prefix} " if prefix else prefix

    edit_distance, lower_edit_distance, simple_edit_distance = distances
    length, length_simple = lengths

    # Calculate CER
    cer = edit_distance_to_cer(edit_distance, length)
    lower_cer = edit_distance_to_cer(lower_edit_distance, length)
    simple_cer = edit_distance_to_cer(simple_edit_distance, length_simple)

    # Print CER stats
    logging.info(f"{prefix}CER        = {cer:.4f} ({edit_distance}/{length})")
    logging.info(f"{prefix}Lower CER  = {lower_cer:.4f} ({lower_edit_distance}"
                 f"/{length})")
    logging.info(f"{prefix}Simple CER = {simple_cer:.4f} "
                 f"({simple_edit_distance}/{length_simple})")
    logging.info("")


def display_statistics(batch_stats: List[Union[int, float]],
                       total_stats: List[Union[int, float]],
                       metrics: List[str]) -> None:
    """
    Logs batch and total statistics for a set of metrics.

    Parameters
    ----------
    batch_stats : List[Union[int, float]]
        A list of batch statistics values, which can be integers or floats.
    total_stats : List[Union[int, float]]
        A list of total statistics values, which can be integers or floats.
    metrics : List[str]
        A list of metric names corresponding to the batch and total statistics.

    Notes
    -----
    This function formats and logs the batch and total statistics for each
    metric. It handles both numeric and textual representations of these
    statistics.
    """

    # Find the maximum length of metric names
    max_metric_length = max(len(metric) for metric in metrics)

    # Prepare headers and format strings
    headers = ["Metric", "Batch", "Total"]
    header_format = "{:>" + str(max_metric_length) + "} | {:>7} | {:>7}"
    row_format = "{:>" + str(max_metric_length) + "} | {:>7} | {:>7}"
    separator = "-" * (max_metric_length + 21)
    border = "=" * (max_metric_length + 21)

    logging.info("Validation metrics:")
    logging.info(border)

    # Print header
    logging.info(header_format.format(*headers))
    logging.info(separator)

    # Print each metric row
    for metric, batch_value, total_value in zip(metrics, batch_stats,
                                                total_stats):
        batch_value_str = f"{batch_value:.4f}" if isinstance(
            batch_value, float) else str(batch_value)
        total_value_str = f"{total_value:.4f}" if isinstance(
            total_value, float) else str(total_value)
        logging.info(row_format.format(
            metric, batch_value_str, total_value_str))

    logging.info(border)


def summarize_model(model: tf.keras.Model) -> List[str]:
    """
    Summarizes a Keras model and returns the summary as a list of strings.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be summarized.

    Returns
    -------
    List[str]
        A list of strings, each containing a line of the model summary.

    Notes
    -----
    This function captures the output of `model.summary()` and returns it as a
    list of strings, making it useful for situations where direct printing of
    the summary is not desirable.
    """

    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))
    return model_layers


# Define this here to avoid circular imports
def edit_distance_to_cer(edit_distance: int, length: int) -> float:
    """
    Calculates the Character Error Rate (CER) based on edit distance and
    length.

    Parameters
    ----------
    edit_distance : int
        The edit distance between the original and predicted texts.
    length : int
        The length of the original text.

    Returns
    -------
    float
        The calculated CER, which is the edit distance divided by the length of
        the original text.

    Notes
    -----
    CER is calculated as the edit distance divided by the maximum of length and
    1, to avoid division by zero.
    """

    return edit_distance / max(length, 1)
