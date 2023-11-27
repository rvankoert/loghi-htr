# Imports

# > Standard library
import logging

# > Local dependencies
from utils import Utils
from model_management import get_prediction_model
from utils import decode_batch_predictions, normalize_confidence


def perform_inference(args, model, inference_dataset, char_list, loader):
    utils_object = Utils(char_list, args.use_mask)
    prediction_model = get_prediction_model(model)

    with open(args.results_file, "w") as results_file:
        for batch_no, batch in enumerate(inference_dataset):
            # Get the predictions
            predictions = prediction_model.predict(batch[0], verbose=0)
            y_pred = decode_batch_predictions(
                predictions, utils_object, args.greedy,
                args.beam_width, args.num_oov_indices)[0]

            # Print the predictions and process the CER
            for index, (confidence, prediction) in enumerate(y_pred):
                # Normalize the confidence before processing because it was
                # determined on the original prediction
                normalized_confidence = normalize_confidence(
                    confidence, prediction)

                # Remove the special characters from the prediction
                prediction = prediction.strip().replace('', '')

                # Format the filename
                filename = loader.get_item(
                    'inference', (batch_no * args.batch_size) + index)

                # Write the results to the results file
                result_str = f"{filename}\t{normalized_confidence}\t" \
                    f"{prediction}"
                logging.info(result_str)
                results_file.write(result_str+"\n")

                # Flush the results file
                results_file.flush()
