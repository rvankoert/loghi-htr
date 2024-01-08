# Imports

# > Standard library
import re
from typing import Tuple

# > Third-party dependencies
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
    set_charlist(chars, use_mask=False, num_oov_indices=0):
        Sets the character list and initializes the StringLookup layers.
    """

    def __init__(self, chars: list, use_mask: bool = False):
        """
        Initializes the Tokenizer with a given character list and mask option.

        Parameters
        ----------
        chars : list
            A list of characters to be used for tokenization.
        use_mask : bool, optional
            A flag to indicate whether to use a mask token (default is False).
        """

        self.set_charlist(chars=chars, use_mask=use_mask)

    def set_charlist(self,
                     chars: list,
                     use_mask: bool = False,
                     num_oov_indices: int = 0):
        """
        Sets the character list and initializes the StringLookup layers.

        Parameters
        ----------
        chars : list
            A list of characters for the tokenizer.
        use_mask : bool, optional
            Whether to include a mask token in the StringLookup layer (default
            is False).
        num_oov_indices : int, optional
            The number of out-of-vocabulary indices (default is 0).

        Raises
        ------
        Exception
            If the character list is empty.
        """

        if not chars:
            raise Exception('No characters found in character list')

        self.charlist = chars
        if num_oov_indices > 0:
            self.charlist = ['[UNK]'] + self.charlist

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
