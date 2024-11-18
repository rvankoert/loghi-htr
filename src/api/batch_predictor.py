# Imports

# > Standard library
import logging
import multiprocessing
import json
import os
import sys
import time
import uuid
from typing import List

# > Third-party dependencies
import tensorflow as tf

# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

# > Local imports
from model.management import load_model_from_directory  # noqa
from setup.environment import initialize_strategy  # noqa


def setup_gpu_environment(gpus: str) -> List[tf.config.PhysicalDevice]:
    """
    Set up the environment for batch prediction.

    Parameters:
    -----------
    gpus : str
        IDs of GPUs to be used (comma-separated).

    Returns:
    --------
    List[tf.config.PhysicalDevice]
        List of active GPUs.
    """
    gpu_devices = tf.config.list_physical_devices('GPU')
    logging.info("Available GPUs: %s", gpu_devices)

    if gpus == "-1":
        active_gpus = []
    elif gpus.lower() == "all":
        active_gpus = gpu_devices
    else:
        gpu_indices = gpus.split(",")
        active_gpus = []
        for i, gpu in enumerate(gpu_devices):
            if str(i) in gpu_indices:
                active_gpus.append(gpu)

    if active_gpus:
        logging.info("Using GPU(s): %s", active_gpus)
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, 'GPU')
    for gpu in active_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return active_gpus


def create_model(base_model_dir: str, model_path: str, strategy: tf.distribute.Strategy) -> tf.keras.Model:
    """
    Load a pre-trained model and cache it.

    Parameters
    ----------
    base_model_dir : str
        Path to the models directory.
    model_path : str
        Path to the pre-trained model file relative to base_model_dir.
    strategy : tf.distribute.Strategy
        Strategy for distributing the model across multiple GPUs.

    Returns
    -------
    tf.keras.Model
        Loaded pre-trained model.
    """
    with strategy.scope():
        try:
            model_location = os.path.join(base_model_dir, model_path)
            model = load_model_from_directory(
                model_location, compile=False)

            # Define the number of channels for the images
            num_channels = update_channels(model_location)

            logging.info("Model %s loaded", model.name)
        except Exception as e:
            logging.error("Error loading model: %s", e)
            raise e
    return model, num_channels


def update_channels(model_path: str) -> int:
    """
    Update the model used for image preparation.

    Parameters
    ----------
    model_path : str
        The path to the directory containing the 'config.json' file.
        The function will append "/config.json" to this path.
    """

    try:
        num_channels = get_model_channels(model_path)
        logging.debug("New number of channels: %s", num_channels)
        return num_channels
    except Exception as e:
        logging.error("Error retrieving number of channels: %s", e)
        raise e


def get_model_channels(config_path: str) -> int:
    """
    Retrieve the number of input channels for a model from a configuration
    file.

    This function reads a JSON configuration file located in the specified
    directory to extract the number of input channels used by the model.

    Parameters
    ----------
    config_path : str
        The path to the directory containing the 'config.json' file.
        The function will append "/config.json" to this path.

    Returns
    -------
    int
        The number of input channels specified in the configuration file.

    Raises
    ------
    FileNotFoundError
        If the 'config.json' file does not exist in the given directory.

    ValueError
        If the number of channels is not found or not specified in the
        'config.json' file.
    """

    config_path = os.path.join(config_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found in the directory: {config_path}")

    # Load the configuration file
    with open(config_path, 'r', encoding='UTF-8') as file:
        config = json.load(file)

    # Extract the number of channels
    # First, check the "model_channels" key, then the "args" key
    num_channels = config.get("model_channels",
                              config.get("args", {}).get("channels", None))
    if num_channels is None:
        raise ValueError(
            "Number of channels not found in the config file.")

    return num_channels


def resize_and_pad_image(image, target_height):
    """
    Resize and pad the image to the desired dimensions.

    Parameters
    ----------
    image : tf.Tensor
        The image tensor.
    target_height : int
        The target height for resizing.

    Returns
    -------
    tf.Tensor
        The resized and padded image tensor.
    """
    shape = tf.shape(image)
    width = tf.cast(shape[1], tf.float32)
    height = tf.cast(shape[0], tf.float32)
    aspect_ratio = width / height
    target_width = tf.cast(target_height * aspect_ratio, tf.int32)
    image = tf.image.resize_with_pad(
        image, target_height, target_width + 50, method=tf.image.ResizeMethod.BILINEAR)

    image = 0.5 - image  # Normalization

    return image


def process_sample(image_bytes, group_id, identifier, model, whitelist, num_channels):
    """
    Preprocess a single sample for the dataset.

    Parameters
    ----------
    image_bytes : tf.Tensor
        The raw image bytes.
    group_id : tf.Tensor
        Group ID.
    identifier : tf.Tensor
        Identifier.
    model : tf.Tensor
        Model path.
    whitelist : tf.Tensor
        Whitelist keys.
    num_channels : int
        Number of channels for the images.

    Returns
    -------
    tuple
        Processed image and associated metadata.
    """

    # Decode and preprocess the image
    try:
        # Decode the image
        image = tf.io.decode_image(
            image_bytes, channels=num_channels, expand_animations=False)
    except tf.errors.InvalidArgumentError:
        # Handle invalid images
        image = tf.zeros([64, 64, num_channels], dtype=tf.float32)
        logging.error("Invalid image for identifier: %s",
                      identifier.numpy().decode('utf-8'))

    # Resize and normalize the image
    image = tf.image.resize(
        image, [64, tf.constant(99999, dtype=tf.int32)], preserve_aspect_ratio=True)
    image = tf.cast(image, tf.float32) / 255.0

    # Resize and pad the image
    image = tf.image.resize_with_pad(
        image, 64, tf.shape(image)[1] + 50, method=tf.image.ResizeMethod.BILINEAR)

    # Normalize the image
    image = 0.5 - image

    # Transpose the image dimensions if necessary
    image = tf.transpose(image, perm=[1, 0, 2])  # From HWC to WHC

    return image, group_id, identifier, model, whitelist


def data_generator(request_queue, current_model_path_holder, stop_event):
    """
    Generator function to yield data from the request queue, ensuring batches contain only one model.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue containing incoming requests.
    current_model_path_holder : list
        A single-element list holding the current model path to allow mutability.

    Yields
    ------
    tuple
        A tuple of image_bytes, group_id, identifier, model, whitelist.
    """
    logging.info("Creating data generator")
    time_since_last_request = time.time()
    has_data = False

    while not stop_event.is_set():
        if not request_queue.empty():
            try:
                time_since_last_request = time.time()
                data = request_queue.get()
                new_model_path = data[3]

                if current_model_path_holder[0] is None:
                    current_model_path_holder[0] = new_model_path

                if new_model_path != current_model_path_holder[0]:
                    # Put the data back for the next generator run
                    request_queue.put(data)
                    logging.info(
                        "Model changed to %s. Switching generator.", new_model_path)
                    # Break to allow the dataset to be recreated with the new model
                    current_model_path_holder[0] = new_model_path
                    break

                has_data = True
                yield data
            except Exception as e:
                logging.error("Error retrieving data from queue: %s", e)
        else:
            time.sleep(0.01)  # Avoid busy waiting

            # Break the loop if no requests for 1 second and yield any remaining data
            if time.time() - time_since_last_request > 1.0 and has_data:
                time_since_last_request = time.time()
                break


def create_dataset(request_queue, batch_size, current_model_path_holder, num_channels, stop_event):
    """
    Create a tf.data.Dataset from the request queue.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue containing incoming requests.
    batch_size : int
        Batch size for the dataset.
    current_model_path_holder : list
        A single-element list holding the current model path.
    num_channels : int
        Number of channels for the images.

    Returns
    -------
    tf.data.Dataset
        The prepared dataset.
    """

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            request_queue, current_model_path_holder, stop_event),
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string),
        output_shapes=((), (), (), (), (None, ))
    )

    # Map the processing function with parallel calls
    dataset = dataset.map(
        lambda image, group_id, identifier, model, metadata: process_sample(
            image, group_id, identifier, model, metadata, num_channels),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # Batch the dataset with padding
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [None, None, num_channels],   # Image shape
            [],                           # group_id
            [],                           # identifier
            [],                           # model
            [None]                        # metadata
        ),
        padding_values=(
            tf.constant(-10, dtype=tf.float32),  # Image padding value
            tf.constant('', dtype=tf.string),    # group_id padding
            tf.constant('', dtype=tf.string),    # identifier padding
            tf.constant('', dtype=tf.string),    # model padding
            tf.constant('', dtype=tf.string)     # metadata padding
        ),
        drop_remainder=False
    )

    # Prefetch data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def batch_prediction_worker(request_queue: multiprocessing.Queue,
                            predicted_queue: multiprocessing.Queue,
                            base_model_dir: str,
                            initial_model_path: str,
                            stop_event: multiprocessing.Event,
                            gpus: str = '0',
                            batch_size: int = 32):
    """
    Worker process for batch prediction on images.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    base_model_dir : str
        Path to the models directory.
    initial_model_path : str
        Initial model path.
    stop_event : multiprocessing.Event
        Event to signal the worker to stop processing.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Default is '0'.
    batch_size : int, optional
        Batch size for predictions.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing status.
    """
    logging.info("Batch prediction process started")

    # Set up the GPU environment and strategy
    active_gpus = setup_gpu_environment(gpus)
    strategy = initialize_strategy(use_float32=False, active_gpus=active_gpus)

    # Initialize model
    current_model_path_holder = [initial_model_path]
    model, num_channels = create_model(base_model_dir,
                                       current_model_path_holder[0],
                                       strategy)

    while not stop_event.is_set():
        # Create the dataset
        dataset = create_dataset(request_queue, batch_size, current_model_path_holder,
                                 num_channels, stop_event)

        # Iterate over the dataset
        for batch in dataset:
            if stop_event.is_set():
                break

            images, batch_groups, batch_identifiers, batch_models, batch_whitelist = batch

            # Check if model needs to be switched
            model_path = batch_models[0].numpy().decode('utf-8')
            if model_path != current_model_path_holder[0]:
                logging.info(
                    "Detected model switch to %s. Reloading model.", current_model_path_holder)
                current_model_path_holder[0] = model_path
                model, num_channels = create_model(
                    base_model_dir, model_path, strategy)

            # Make predictions
            with strategy.scope():
                logging.info("Making predictions for batch")
                encoded_predictions = model.predict_on_batch(images)
                logging.info("Predictions made")

            batch_id = uuid.uuid4()
            predicted_queue.put((encoded_predictions, batch_groups, batch_identifiers,
                                 model_path, batch_id, batch_whitelist))

    logging.info("Batch prediction process stopped")
