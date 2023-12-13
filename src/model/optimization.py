# Imports

# > Standard library
from typing import Union

# Third-party dependencies
import tensorflow as tf


class CustomLearningRateSchedule(tf.keras.optimizers.
                                 schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate: float, decay_rate: float,
                 decay_steps: int, warmup_ratio: float, total_steps: int,
                 decay_per_step: bool = True) -> None:
        """
        Initialize the custom learning rate schedule.

        Parameters
        ----------
        initial_learning_rate : float
            The initial learning rate.
        decay_rate : float
            The rate at which the learning rate decays.
        decay_steps : int
            The number of steps after which the learning rate decays. If -1,
            uses `train_batches`.
        warmup_ratio : float
            The ratio of the warmup period with respect to the total training
            steps.
        total_steps : int
            Total number of training steps.
        decay_per_step : bool, optional
            If True, apply decay per step; otherwise, apply per epoch (default
            is True).
        """

        super(CustomLearningRateSchedule, self).__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate,
                                             tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)

        # Use the total number of training batches as decay steps if
        # decay_steps is -1
        self.decay_steps = tf.cast(decay_steps, tf.float32)

        # Calculate warmup steps as a fraction of total training steps
        self.warmup_steps = tf.cast(warmup_ratio * total_steps, tf.float32)
        self.decay_per_step = tf.cast(decay_per_step, tf.bool)

        self.linear_decay = tf.constant(True, dtype=tf.bool)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step: tf.Tensor) -> float:
        """
        Calculate the learning rate for a given step.

        Parameters
        ----------
        step : tf.Tensor
            The current training step.

        Returns
        -------
        float
            The calculated learning rate for the given step.
        """
        # Ensure `step` is a float tensor for division
        step = tf.cast(step, tf.float32)

        def warmup_lr():
            return self.initial_learning_rate * (step / self.warmup_steps)

        def linear_decayed_lr():
            # Post-warmup phase
            def per_step():
                # Calculate the proportion of steps completed
                proportion_completed = (
                    step - self.warmup_steps) / self.total_steps

                return self.initial_learning_rate * (1 - proportion_completed)

            def per_epoch():
                epoch = tf.math.floor(step / self.decay_steps)
                total_epochs = tf.math.floor(self.total_steps /
                                             self.decay_steps)

                # Calculate the proportion of epochs completed
                proportion_completed = epoch / total_epochs

                return self.initial_learning_rate * (1 - proportion_completed)

            return tf.cond(self.decay_per_step, per_step, per_epoch)

        def exponential_decayed_lr():
            # Post-warmup phase
            def per_step():
                steps_since_warmup = step - self.warmup_steps
                return self.initial_learning_rate * tf.pow(
                    self.decay_rate, steps_since_warmup / self.decay_steps)

            def per_epoch():
                return self.initial_learning_rate * tf.pow(
                    self.decay_rate, tf.math.floor(step / self.decay_steps))

            return tf.cond(self.decay_per_step, per_step, per_epoch)

        # Use tf.cond to choose between warmup and decay phase
        return tf.cond(step < self.warmup_steps, warmup_lr,
                       lambda: tf.cond(self.linear_decay, linear_decayed_lr,
                                       exponential_decayed_lr))

    def get_config(self) -> dict:
        """
        Return the configuration of the learning rate schedule.

        Returns
        -------
        dict
            A dictionary containing the configuration parameters.
        """

        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "decay_per_step": self.decay_per_step
        }


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
                                  train_batches: int, do_train: bool,
                                  warmup_ratio: float, epochs: int,
                                  decay_per_step: bool = False) \
        -> Union[float, CustomLearningRateSchedule]:
    """
    Creates a learning rate schedule based on the specified parameters,
    with support for warmup and custom decay.

    Parameters
    ----------
    learning_rate : float
        The initial learning rate.
    decay_rate : float
        The rate of decay for the learning rate.
    train_batches : int
        The total number of training batches.
    do_train : bool
        Indicates whether training is being performed.
    warmup_ratio : float, optional
        The ratio of the warmup period to the total training batches (default
        is 0.1).
    epochs : int
        The total number of epochs.
    decay_per_step : bool, optional
        If True, apply decay per step; otherwise, apply per epoch (default is
        False).

    Returns
    -------
    Union[float, CustomLearningRateSchedule]
        The learning rate or a learning rate schedule based on the provided
        parameters.
    """

    if do_train:
        # Use custom learning rate schedule with warmup and decay
        return CustomLearningRateSchedule(
            initial_learning_rate=learning_rate,
            decay_rate=decay_rate,
            decay_steps=train_batches,
            warmup_ratio=warmup_ratio,
            total_steps=epochs * train_batches + 1,
            decay_per_step=decay_per_step
        )
    else:
        # If not training, return the initial learning rate
        return learning_rate
