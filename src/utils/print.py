# Imports

# > Standard library
import logging


def print_predictions(filename, original_text, predicted_text, char_str=None):
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info(f"File: {filename}")
    logging.info("")
    logging.info(f"Original text  - {original_text}")
    logging.info(f"Predicted text - {predicted_text}")
    if char_str:
        logging.info(f"WordBeamSearch - {char_str}")
    logging.info("")


def print_cer_stats(distances, lengths, prefix=""):
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


def display_statistics(batch_stats, total_stats, metrics):
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


def summarize_model(model):
    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))
    return model_layers


# Define this here to avoid circular imports
def edit_distance_to_cer(edit_distance, length):
    return edit_distance / max(length, 1)
