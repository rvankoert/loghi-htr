# Imports

# > Standard library
from typing import Union

# Third-party dependencies
import tensorflow as tf


def get_optimizer(optimizer_name: str,
                  learning_rate_schedule: Union[float,
                                                tf.keras.optimizers.schedules
                                                .LearningRateSchedule]) \
        -> tf.keras.optimizers.Optimizer:
    """
    Selects and creates an optimizer based on the provided optimizer name and
    learning rate schedule.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer to be created.
    learning_rate_schedule : Union[float,
                                   tf.keras.optimizers.schedules.LearningRateSchedule]
        The learning rate or learning rate schedule to be used with the
        optimizer.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        An instance of the specified optimizer with the given learning rate
        schedule.

    Raises
    ------
    ValueError
        If the optimizer name is not recognized.
    """

    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "adamw": tf.keras.optimizers.experimental.AdamW,
        "adadelta": tf.keras.optimizers.Adadelta,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adamax": tf.keras.optimizers.Adamax,
        "adafactor": tf.keras.optimizers.Adafactor,
        "nadam": tf.keras.optimizers.Nadam
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name](learning_rate=learning_rate_schedule)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def create_learning_rate_schedule(learning_rate: float, decay_rate: float,
                                  decay_steps: int, train_batches: int,
                                  do_train: bool) \
        -> Union[float, tf.keras.optimizers.schedules.ExponentialDecay]:
    """
    Creates a simple learning rate schedule based on the specified parameters.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    decay_rate : float
        The rate of decay for the learning rate.
    decay_steps : int
        The number of steps after which the learning rate decays. A value of -1
        indicates using the total number of training batches as the decay
        steps.
    train_batches : int
        The total number of training batches.
    do_train : bool
        Indicates whether training is being performed.

    Returns
    -------
    Union[float, tf.keras.optimizers.schedules.ExponentialDecay]
        The learning rate if no decay is applied, or an exponential decay
        learning rate schedule based on the provided parameters.

    Notes
    -----
    If `decay_rate` is 0 or `do_train` is False, the method returns the initial
    `learning_rate` without applying any decay.
    """

    if decay_rate > 0 and do_train:
        if decay_steps > 0:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate
            )
        elif decay_steps == -1:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=train_batches,
                decay_rate=decay_rate
            )
    return learning_rate
