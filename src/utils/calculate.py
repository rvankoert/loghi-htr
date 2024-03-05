# Imports

# > Standard library
from typing import Dict, List, Tuple, Union

# > Third-party dependencies
import editdistance

# > Local imports
from utils.text import simplify_text
from utils.print import print_cer_stats


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
    """

    return edit_distance / max(length, 1)


def calc_95_confidence_interval(cer_metric: float, n: int) -> float:
    """
    Calculates the 95% confidence interval for a given CER metric based on the
    number of samples.

    Parameters
    ----------
    cer_metric : float
        The CER metric for which the confidence interval is to be calculated.
    n : int
        The number of samples used in the calculation of the CER metric.

    Returns
    -------
    float
        The 95% confidence interval for the given CER metric.
    """

    return 1.96 * ((cer_metric*(1-cer_metric))/n) ** 0.5


def calculate_edit_distances(prediction: str, original_text: str) \
        -> Tuple[int, int, int]:
    """
    Calculates the edit distances between the predicted text and the original
    text in various forms.

    Parameters
    ----------
    prediction : str
        The text predicted by the model.
    original_text : str
        The original text.

    Returns
    -------
    Tuple[int, int, int]
        A tuple containing the standard edit distance, the edit distance for
        lower-cased texts, and the edit distance for simplified texts
        (alphanumeric only).

    Notes
    -----
    This function preprocesses both the original and predicted texts into
    lower-cased and simplified forms before calculating the respective edit
    distances.
    """

    # Preprocess the text
    lower_prediction, simple_prediction = simplify_text(prediction)
    lower_original, simple_original = simplify_text(original_text)

    # Calculate edit distance
    edit_distance = editdistance.eval(prediction, original_text)
    lower_edit_distance = editdistance.eval(lower_prediction,
                                            lower_original)
    simple_edit_distance = editdistance.eval(simple_prediction,
                                             simple_original)

    return edit_distance, lower_edit_distance, simple_edit_distance


def calculate_cers(info: Dict[str, int], prefix: str = "") \
        -> Tuple[float, float, float]:
    """
    Calculates the Character Error Rates (CER) for standard, lower-cased, and
    simplified texts.

    Parameters
    ----------
    info : Dict[str, int]
        A dictionary containing the edit distances and lengths for the
        different text forms.
    prefix : str, optional
        A prefix to identify the relevant keys in the dictionary (e.g.,
        'Normalized' for normalized texts).

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the CER for standard, lower-cased, and simplified
        texts.

    Notes
    -----
    This function uses the 'edit_distance_to_cer' function to calculate the
    CERs based on the information provided in the 'info' dictionary.
    """

    edit_distance = info[prefix + 'edit_distance']
    length = info[prefix + 'length']
    lower_edit_distance = info[prefix + 'lower_edit_distance']
    length_simple = info[prefix + 'length_simple']
    simple_edit_distance = info[prefix + 'simple_edit_distance']

    # Calculate CER
    cer = edit_distance_to_cer(edit_distance, length)
    lower_cer = edit_distance_to_cer(lower_edit_distance, length)
    simple_cer = edit_distance_to_cer(simple_edit_distance, length_simple)

    return cer, lower_cer, simple_cer


# Prediction processing functions

def update_statistics(batch_info: Dict[str, int],
                      total_counter: Dict[str, int]) \
        -> Tuple[List[str], List[Union[int, float]], List[Union[int, float]]]:
    """
    Processes and updates CER statistics for a batch and the overall totals.

    Parameters
    ----------
    batch_info : Dict[str, int]
        The information about the current batch, including edit distances and
        lengths.
    total_counter : Dict[str, int]
        The running totals of edit distances and lengths.

    Returns
    -------
    Tuple[List[str], List[Union[int, float]], List[Union[int, float]]]
        A tuple containing updated total counters, metrics, batch statistics,
        and total statistics.

    Notes
    -----
    This function is used to calculate and update the CERs for both the current
    batch and overall totals, and it updates the metrics and statistics lists
    accordingly.
    """

    metrics, batch_stats, total_stats = [], [], []
    keys = list(batch_info.keys())

    # Iterate over the keys in the batch info
    # We take every 5th key because the batch info contains 3 edit distances
    # and 2 lengths for each text form
    for i in range(0, len(batch_info), 5):
        split_key = keys[i].split(" ")
        prefix = split_key[0] + " " if len(split_key) > 1 else ""

        # Calculate CERs for both batch and total
        batch_cers = calculate_cers(batch_info, prefix=prefix)
        total_cers = calculate_cers(total_counter, prefix=prefix)

        # Define metric names based on the prefix
        cer_names = [f"{prefix}CER", f"{prefix}Lower CER",
                     f"{prefix}Simple CER"]

        # Extend metrics and stats
        metrics.extend(cer_names)
        batch_stats.extend(batch_cers)
        total_stats.extend(total_cers)

    return metrics, batch_stats, total_stats


def increment_counters(distances: Dict[str, Tuple[int, int, int]],
                       originals: Dict[str, str],
                       batch_counter: Dict[str, int],
                       do_print: bool) \
        -> Dict[str, int]:
    """
    Processes a single prediction by calculating edit distances and updating
    batch information.

    Parameters
    ----------
    distances : Dict[str, Tuple[int, int, int]]
        A dictionary containing the edit distances for standard, lower-cased,
        and simplified texts.
    originals : Dict[str, str]
        A dictionary containing the original texts for standard, lower-cased,
        and simplified texts.
    batch_counter : Dict[str, int]
        The dictionary to update with the new edit distances and lengths.
    do_print : bool
        A flag indicating whether to print the CER statistics for this
        prediction.

    Returns
    -------
    Dict[str, int]
        The updated batch information with new edit distances and lengths.

    Notes
    -----
    This function calculates edit distances for various text forms, prints CER
    statistics if requested, and updates the batch information.
    """

    for prefix in ["", "Normalized ", "WBS "]:
        # Get the distances and original text
        distance = distances[prefix + 'distances']
        original = originals[prefix + 'original']

        # Update the batch counter
        if distance is not None:
            # Preprocess the text for CER calculation
            _, simple_original = simplify_text(original)

            # Determine the lengths
            lengths = [len(original), len(simple_original)]

            # Unpack the distances
            edit_distance, lower_edit_distance, simple_edit_distance = distance

            # Print the predictions if there are any errors
            if do_print:
                print_cer_stats(distance, lengths, prefix=prefix)

            # Update the counters
            batch_counter[prefix + 'edit_distance'] \
                += edit_distance
            batch_counter[prefix + 'length'] \
                += lengths[0]
            batch_counter[prefix + 'lower_edit_distance'] \
                += lower_edit_distance
            batch_counter[prefix + 'length_simple'] \
                += lengths[1]
            batch_counter[prefix + 'simple_edit_distance'] \
                += simple_edit_distance

    return batch_counter
