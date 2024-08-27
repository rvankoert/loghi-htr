# Imports

# > Standard library
import json
import re
from typing import Tuple, Union

# > Third-party dependencies
import numpy as np
import tensorflow as tf


class Tokenizer:
    """
    A tokenizer class for character-based tokenization.
    This class provides methods for converting a list of characters into
    a TensorFlow StringLookup layer, which maps characters to integers and
    vice versa. It supports out-of-vocabulary (OOV) tokens and allows saving
    and loading the tokenizer configuration to/from a JSON file.
    """

    def __init__(self,
                 chars: list = None,
                 num_oov_indices: int = 1,
                 json_path: str = None):
        """
        Initializes the Tokenizer with a given character list or loads from a JSON file if provided.

        Parameters
        ----------
        chars : list, optional
            A list of characters to be used for tokenization.
        num_oov_indices : int, optional
            The number of out-of-vocabulary indices (default is 1).
        json_path : str, optional
            Path to a JSON file to load the tokenizer configuration.
        """
        if json_path:
            self.load_from_json(json_path)
        else:
            if chars is None or not chars:
                raise ValueError("The character list cannot be empty.")
            self.charlist = list(chars)
            self.num_oov_indices = num_oov_indices
            self._initialize_string_lookup_layers()

    def _initialize_string_lookup_layers(self):
        """Initializes the StringLookup layers."""
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=self.charlist,
            num_oov_indices=self.num_oov_indices,
            oov_token='[UNK]',
            encoding="UTF-8"
        )
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(),
            num_oov_indices=0,
            oov_token='',
            encoding="UTF-8",
            invert=True
        )

    def __call__(self, texts: Union[str, list]) -> tf.Tensor:
        """
        Tokenizes the input text(s) into a sequence of integers.

        Parameters
        ----------
        texts : str or list of str
            The text or a list of texts to be tokenized.

        Returns
        -------
        tf.Tensor
            A tensor of tokenized integer sequences.
        """
        return self.char_to_num(tf.strings.unicode_split(texts, 'UTF-8'))

    def __len__(self):
        """[TODO:description]

        Returns
        -------
        [TODO:return]
            [TODO:description]

        """
        return len(self.charlist)

    def __str__(self):
        vocab = self.char_to_num.get_vocabulary()
        data = {i: char for i, char in enumerate(vocab)}
        return json.dumps(data, ensure_ascii=False, indent=4)

    def encode(self, texts: Union[str, list]) -> tf.Tensor:
        """
        Encodes the input text(s) into a sequence of integers.

        Parameters
        ----------
        texts : str or list of str
            The text or a list of texts to be encoded.

        Returns
        -------
        tf.Tensor
            A tensor of encoded integer sequences.
        """
        return self(texts)

    def decode(self, tokenized_texts: Union[tf.Tensor, list]) -> tf.Tensor:
        """
        Decodes the tokenized sequences back into text.

        Parameters
        ----------
        tokenized_texts : tf.Tensor or list of tf.Tensor
            The tokenized integer sequences or a list of sequences.

        Returns
        -------
        tf.Tensor
            A tensor of decoded strings.
        """
        decoded = tf.strings.reduce_join(self.num_to_char(tokenized_texts),
                                         axis=-1).numpy()
        if isinstance(decoded, bytes):
            return decoded.decode("utf-8")
        if isinstance(decoded, np.ndarray):
            return np.array([d.decode("utf-8") for d in decoded])
        return decoded

    def save_to_json(self, json_path: str):
        """
        Saves the tokenizer's vocabulary to a JSON file.

        Parameters
        ----------
        json_path : str
            The file path to save the JSON file.
        """
        vocab = self.char_to_num.get_vocabulary()
        data = {i: char for i, char in enumerate(vocab)}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_from_json(self, json_path: str):
        """
        Loads the tokenizer's vocabulary from a JSON file.

        Parameters
        ----------
        json_path : str
            The file path to the JSON file.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        if json_path.endswith('.txt'):
            # Convert TXT to JSON
            logging.warning("TXT file detected. Converting to JSON format...")
            with open(json_path, 'r', encoding='utf-8') as f:
                chars = f.read().splitlines()
            data = {i: char for i, char in enumerate(chars)}
            json_path = json_path.replace('.txt', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Converted and saved as JSON: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chars = [data[str(i)] for i in range(len(data))]
        self.charlist = chars
        self.num_oov_indices = 1
        self._initialize_string_lookup_layers()

    def add_tokens(self, tokens: Union[str, list]):
        """
        Adds a token or a list of tokens to the tokenizer.

        Parameters
        ----------
        tokens : str or list of str
            The token or a list of tokens to be added to the tokenizer.
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        # Add new tokens to the character list if they don't already exist
        new_tokens = [token for token in tokens if token not in self.charlist]

        if new_tokens:
            self.charlist.extend(new_tokens)
            self._initialize_string_lookup_layers()


def remove_tags(text: str) -> str:
    """
    Removes specific control characters from the given text.

    Parameters
    ----------
    text : str
        The text from which control characters need to be removed.

    Returns
    -------
    str
        The text with specified control characters removed.

    Notes
    -----
    This function specifically targets the removal of control characters like
    '␃', '␅', '␄', and '␆' from the text.
    """

    return re.sub(r'[␃␅␄␆]', '', text)


def preprocess_text(text: str) -> str:
    """
    Processes the given text by stripping, replacing certain characters, and
    removing tags.

    Parameters
    ----------
    text : str
        The text to be processed.

    Returns
    -------
    str
        The preprocessed text.

    Notes
    -----
    This function performs operations like stripping whitespace, replacing
    specific characters (e.g., ''), and removing certain control tags using
    the `remove_tags` function.
    """

    text = text.strip().replace('', '')
    text = remove_tags(text)
    return text


def simplify_text(text: str) -> Tuple[str, str]:
    """
    Simplifies the given text by converting it to lowercase and removing
    non-alphanumeric characters.

    Parameters
    ----------
    text : str
        The text to be simplified.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the lowercase version of the original text and the
        simplified text with only alphanumeric characters.

    Notes
    -----
    This function first converts the text to lowercase and then further
    simplifies it by removing all characters except basic alphanumeric (letters
    and numbers).
    """

    lower_text = text.lower()
    simple_text = re.sub(r'[^a-zA-Z0-9]', '', lower_text)
    return lower_text, simple_text


def normalize_text(input: str, replacements: str) -> str:
    """
    Normalize text using a json file with replacements

    Parameters
    ----------
    input : str
        Input string to normalize
    replacements : str
        Path to json file with replacements, where key is the string to
        replace and value is the replacement. Example: {"a": "b"} will
        replace all "a" with "b" in the input string.

    Returns
    -------
    str
        Normalized string
    """

    with open(replacements, 'r') as f:
        replacements = json.load(f)
        for key, value in replacements.items():
            input = input.replace(key, value)

    input = re.sub(r"\s+", " ", input)

    return input.strip()
