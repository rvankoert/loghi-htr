# Imports

# > Standard library
import os
import json
import re
from typing import Tuple, Union, List
import logging

# > Third-party dependencies
import numpy as np
import tensorflow as tf


class Tokenizer:
    """
    A tokenizer class for token-based tokenization.
    This class provides methods for converting a list of tokens (can be multi-length)
    into a TensorFlow StringLookup layer, which maps tokens to integers and vice versa.
    It supports out-of-vocabulary (OOV) tokens and allows saving and loading
    the tokenizer configuration to/from a JSON file.
    """

    def __init__(self,
                 tokens: List[str] = None):
        """
        Initializes a new Tokenizer with a list of tokens.

        Parameters
        ----------
        tokens : list, optional
            A list of tokens to be used for tokenization.
        """
        if tokens is None or not tokens:
            raise ValueError("The token list cannot be empty.")
        self.token_list = tokens
        self._initialize_string_lookup_layers()

    def _initialize_string_lookup_layers(self):
        """Initializes the StringLookup layers for tokens."""
        self.token_to_num = tf.keras.layers.StringLookup(
            vocabulary=self.token_list,
            num_oov_indices=1,
            oov_token='[UNK]',
            encoding="UTF-8",
            mask_token='[PAD]'
        )
        self.num_to_token = tf.keras.layers.StringLookup(
            vocabulary=self.token_to_num.get_vocabulary(),
            num_oov_indices=0,
            oov_token='',
            encoding="UTF-8",
            invert=True,
            mask_token='[PAD]'
        )

        self.token_list = self.token_to_num.get_vocabulary()

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Class method to load the tokenizer's vocabulary from a JSON file or a legacy
        charlist.txt file.

        If a charlist.txt file is provided, it loads the character list from a single line,
        converts it to a new tokenizer.json file, and saves it in the same directory.
        Skips specific unwanted Unicode characters during this process.

        Parameters
        ----------
        file_path : str
            The file path to the JSON file or legacy charlist.txt file.

        Returns
        -------
        Tokenizer
            An instance of the Tokenizer class initialized with the loaded tokens.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith('.txt'):
            logging.warning("Loading from a legacy charlist.txt file. "
                            "This file will be converted to a new tokenizer.json file.")
            # Handle legacy charlist.txt file
            unwanted_chars = ["", ""]  # pylint: disable=E2513
            with open(file_path, 'r', encoding='utf-8') as f:
                # Convert the single line of characters into a list of characters
                chars = [char for char in f.read() if not
                         char in unwanted_chars]

            # Save the tokens as a new tokenizer.json file
            json_dir = os.path.dirname(file_path)
            json_file = os.path.join(json_dir, 'tokenizer.json')

            # Create a new instance of Tokenizer
            tokenizer = cls(tokens=chars)
            tokenizer.save_to_json(json_file)

            logging.info("Legacy charlist.txt file loaded and converted to %s",
                         json_file)
            return tokenizer

        # Load from JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokens = [data[str(i)] for i in range(len(data))]

        logging.info("Tokenizer loaded from %s", file_path)
        return cls(tokens=tokens)

    def __call__(self, texts: Union[str, List[str]]) -> tf.Tensor:
        """
        Tokenizes the input text(s) into a sequence of token indices.

        This method expects a string or list of strings and tokenizes them
        into a sequence of integers based on the tokenizer's vocabulary.

        Parameters
        ----------
        texts : str or list of str
            The text or a list of texts to be tokenized.

        Returns
        -------
        tf.Tensor
            A tensor of tokenized integer sequences.
        """
        if isinstance(texts, str):
            texts = [texts]

        split_texts = tf.strings.unicode_split(texts, 'UTF-8')
        return self.token_to_num(split_texts)

    def __len__(self):
        """Returns the number of tokens in the tokenizer's vocabulary."""
        return len(self.token_to_num.get_vocabulary())

    def __str__(self):
        """Returns a JSON-formatted string representation of the tokenizer's vocabulary."""
        vocab = self.token_to_num.get_vocabulary()
        data = dict(enumerate(vocab))
        return json.dumps(data, ensure_ascii=False, indent=4)

    def encode(self, texts: Union[str, List[str]]) -> tf.Tensor:
        """
        Encodes the input text(s) into a sequence of token indices.

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

    def decode(self, tokenized_texts: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
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
        decoded = tf.strings.reduce_join(self.num_to_token(tokenized_texts),
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
        vocab = self.token_to_num.get_vocabulary()
        data = dict(enumerate(vocab))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def add_tokens(self, tokens: Union[str, List[str]]):
        """
        Adds a token or a list of tokens to the tokenizer.

        Parameters
        ----------
        tokens : str or list of str
            The token or a list of tokens to be added to the tokenizer.
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        # Add new tokens to the token list if they don't already exist
        new_tokens = [
            token for token in tokens if token not in self.token_list]

        if new_tokens:
            self.token_list.extend(new_tokens)
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
    specific characters (e.g., '[PAD]'), and removing certain control tags using
    the `remove_tags` function.
    """

    text = text.strip().replace('[PAD]', '')
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


def normalize_text(text: str, replacements: str) -> str:
    """
    Normalize text using a json file with replacements

    Parameters
    ----------
    text : str
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

    with open(replacements, 'r', encoding='utf-8') as f:
        replacements = json.load(f)
        for key, value in replacements.items():
            text = text.replace(key, value)

    text = re.sub(r"\s+", " ", text)

    return text.strip()
