from enum import Enum


class ArgumentType(Enum):
    """
    Enum to define the types of arguments that can be passed to the application.
    """
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SINGLE_FILE_PATH = "single_file_path"
    MULTIPLE_FILE_PATH = "multiple_file_path"
    DIRECTORY_PATH = "directory_path"


class Argument:
    """
    Class to handle command line arguments for the application.
    """

    def __init__(self, name, default=None, type=None, help=None):
        self.name = name
        self.default = default
        self.type = type
        self.help = help
        self.min_value = None
        self.max_value = None

    def __str__(self):
        return f"argument(name={self.name}, default={self.default}, type={self.type}, help={self.help})"

# list of arguments
arguments = [
    Argument(
        name="--model",
        default="model.h5",
        type=ArgumentType.STRING,
        help="Path to the model file."
    ),
    Argument(
        name="--corpus_file",
        default=None,
        type=ArgumentType.STRING,
        help="Path to the corpus file."
    ),
    Argument(
        name="--normalization_file",
        default=None,
        type=ArgumentType.STRING,
        help="Path to the normalization file."
    ),
    Argument(
        name="--output_dir",
        default="./output/",
        type=ArgumentType.STRING,
        help="Directory to save the output files."
    ),
]