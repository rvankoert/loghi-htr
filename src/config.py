# import the necessary packages
import os


class config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # specify the shape of the inputs for our network
    IMG_SHAPE = (128, 128, 4)
    # specify the batch size and number of epochs
    BATCH_SIZE = 1
    EPOCHS = 1
    NUM_CHANNELS = 3

    SEED = 42
    GPU = 0
    LEARNING_RATE = 0.001
    SPEC = 'Cl11,11,32 Mp3,3 Cl7,7,64 Gm'
    # define the path to the base output directory
    BASE_OUTPUT = "output"
    MODEL_NAME = "difornet10"
    LOSS = ""
    OPTIMIZER = ""
