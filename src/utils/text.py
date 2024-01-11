# Imports

# > Standard library
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
    vice versa. It supports out-of-vocabulary (OOV) tokens and optional
    masking.

    Attributes
    ----------
    charList : list
        A list of characters to be used for tokenization.
    char_to_num : tf.keras.layers.StringLookup
        A TensorFlow StringLookup layer mapping characters to integers.
    num_to_char : tf.keras.layers.StringLookup
        A TensorFlow StringLookup layer mapping integers back to characters.

    Methods
    -------
    __call__(texts):
        Tokenizes the input text(s) into a sequence of integers.
    encode(texts):
        Encodes the input text(s) into a sequence of integers.
    decode(tokenized_texts):
        Decodes the tokenized sequences back into text.
    """

    def __init__(self, chars: list, use_mask: bool = False,
                 num_oov_indices: int = 0):
        """
        Initializes the Tokenizer with a given character list and mask option.

        Parameters
        ----------
        chars : list
            A list of characters to be used for tokenization.
        use_mask : bool, optional
            A flag to indicate whether to use a mask token (default is False).
        num_oov_indices : int, optional
            The number of out-of-vocabulary indices (default is 0).

        Raises
        ------
        ValueError
            If the character list is empty.
        """

        self.charlist = list(chars)
        if not self.charlist:
            raise ValueError("The character list cannot be empty.")

        mask_token = '' if use_mask else None
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=self.charlist,
            num_oov_indices=num_oov_indices,
            mask_token=mask_token,
            oov_token='[UNK]',
            encoding="UTF-8"
        )

        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(),
            num_oov_indices=0,
            oov_token='',
            mask_token=mask_token,
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
        elif isinstance(decoded, np.ndarray):
            return np.array([d.decode("utf-8") for d in decoded])
        else:
            return decoded


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
