# Imports

# > Local imports
from utils.text import simplify_text
from utils.print import print_cer_stats

# > Third-party dependencies
import editdistance


def edit_distance_to_cer(edit_distance, length):
    return edit_distance / max(length, 1)


def calc_95_confidence_interval(cer_metric, n):
    """ Calculates the binomial confidence radius of the given metric
    based on the num of samples (n) and a 95% certainty number
    E.g. cer_metric = 0.10, certainty = 95 and n= 5500 samples -->
    conf_radius = 1.96 * ((0.1*(1-0.1))/5500)) ** 0.5 = 0.008315576
    This means with 95% certainty we can say that the True CER of the model is
    between 0.0917 and 0.1083 (4-dec rounded)
    """
    return 1.96 * ((cer_metric*(1-cer_metric))/n) ** 0.5


def calculate_edit_distances(prediction, original_text):
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


def calculate_cers(info, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix

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


def update_totals(info, total, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix

    edit_distance = info[prefix + 'edit_distance']
    length = info[prefix + 'length']
    lower_edit_distance = info[prefix + 'lower_edit_distance']
    length_simple = info[prefix + 'length_simple']
    simple_edit_distance = info[prefix + 'simple_edit_distance']

    total[prefix + 'edit_distance'] += edit_distance
    total[prefix + 'length'] += length
    total[prefix + 'lower_edit_distance'] += lower_edit_distance
    total[prefix + 'length_simple'] += length_simple
    total[prefix + 'simple_edit_distance'] += simple_edit_distance

    return total


def update_batch_info(info, distances, lengths, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix
    edit_distance, lower_edit_distance, simple_edit_distance = distances
    length, length_simple = lengths

    info[f'{prefix}edit_distance'] += edit_distance
    info[f'{prefix}length'] += length
    info[f'{prefix}lower_edit_distance'] += lower_edit_distance
    info[f'{prefix}length_simple'] += length_simple
    info[f'{prefix}simple_edit_distance'] += simple_edit_distance

    return info


# Prediction processing functions

def process_cer_type(batch_info, total_counter, metrics, batch_stats,
                     total_stats, prefix=""):
    # Update totals
    updated_totals = update_totals(batch_info, total_counter, prefix=prefix)

    # Calculate CERs for both batch and total
    batch_cers = calculate_cers(batch_info, prefix=prefix)
    total_cers = calculate_cers(updated_totals, prefix=prefix)

    # Define metric names based on the prefix
    prefix = f"{prefix} " if prefix else prefix
    cer_names = [f"{prefix}CER", f"{prefix}Lower CER", f"{prefix}Simple CER"] \
        if prefix else ["CER", "Lower CER", "Simple CER"]

    # Extend metrics and stats
    metrics.extend(cer_names)
    batch_stats.extend(batch_cers)
    total_stats.extend(total_cers)

    return updated_totals, metrics, batch_stats, total_stats


def process_prediction_type(prediction, original, batch_info,
                            do_print, prefix=""):
    # Preprocess the text for CER calculation
    _, simple_original = simplify_text(original)

    # Calculate edit distances
    distances = calculate_edit_distances(prediction, original)

    # Unpack the distances
    edit_distance, lower_edit_distance, simple_edit_distance = distances
    lengths = [len(original), len(simple_original)]

    # Print the predictions if there are any errors
    if do_print:
        print_cer_stats(distances, lengths, prefix=prefix)

    # Update the counters
    batch_info = update_batch_info(batch_info,
                                   distances,
                                   lengths,
                                   prefix=prefix)

    return batch_info


def calculate_confidence_intervals(cer_metrics, n):
    intervals = []

    # Calculate the confidence intervals
    for cer_metric in cer_metrics:
        intervals.append(calc_95_confidence_interval(cer_metric, n))

    return intervals
