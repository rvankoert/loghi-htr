# Imports

# > Standard library
from typing import Union

# Third-party dependencies
import tensorflow as tf


class CustomLearningRateSchedule(tf.keras.optimizers.
                                 schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate: float, decay_rate: float,
                 decay_steps: int, warmup_ratio: float, train_batches: int,
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
        train_batches : int
            Total number of training batches.
        decay_per_step : bool, optional
            If True, apply decay per step; otherwise, apply per epoch (default
            is True).
        """

        super(CustomLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

        # Use the total number of training batches as decay steps if
        # decay_steps is -1
        self.decay_steps = decay_steps if decay_steps != -1 else train_batches

        # Calculate warmup steps as a fraction of total training steps
        self.warmup_steps = int(warmup_ratio * train_batches)
        self.decay_per_step = decay_per_step

    def __call__(self, step: int) -> float:
        """
        Calculate the learning rate for a given step.

        Parameters
        ----------
        step : int
            The current training step.

        Returns
        -------
        float
            The calculated learning rate for the given step.
        """
        # Warmup phase: linearly increase learning rate
        if step < self.warmup_steps:
            warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
            return warmup_lr
        else:
            # Post-warmup phase: apply exponential decay
            if self.decay_per_step:
                # Decay per step
                decayed_lr = self.initial_learning_rate * \
                    (self.decay_rate ** (step / self.decay_steps))
            else:
                # Decay per epoch
                decayed_lr = self.initial_learning_rate * \
                    (self.decay_rate ** (step // self.decay_steps))
            return decayed_lr

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
            "train_batches": self.train_batches,
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
                                  decay_steps: int, train_batches: int,
                                  do_train: bool, warmup_ratio: float = 0.1,
                                  decay_per_step: bool = False) \
        -> Union[float, CustomLearningRateSchedule,
                 tf.keras.optimizers.schedules.ExponentialDecay]:
    """
    Creates a learning rate schedule based on the specified parameters,
    with support for warmup and custom decay.

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
    warmup_ratio : float, optional
        The ratio of the warmup period to the total training batches (default
        is 0.1).
    decay_per_step : bool, optional
        If True, apply decay per step; otherwise, apply per epoch (default is
        False).

    Returns
    -------
    Union[float, CustomLearningRateSchedule,
          tf.keras.optimizers.schedules.ExponentialDecay]
        The learning rate or a learning rate schedule based on the provided
        parameters.
    """

    if do_train:
        if decay_rate > 0:
            # Use custom learning rate schedule with warmup and decay
            return CustomLearningRateSchedule(
                initial_learning_rate=learning_rate,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                warmup_ratio=warmup_ratio,
                train_batches=train_batches,
                decay_per_step=decay_per_step
            )
        else:
            # Return a constant learning rate
            return learning_rate
    else:
        # If not training, return the initial learning rate
        return learning_rate
