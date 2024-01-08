# Imports

# > Standard library
import re
from typing import Tuple

# > Third-party dependencies
import tensorflow as tf


class Tokenizer:
    def __init__(self, chars, use_mask):
        self.set_charlist(chars=chars, use_mask=use_mask)

    def set_charlist(self, chars, use_mask=False, num_oov_indices=0):
        self.charList = chars
        if num_oov_indices > 0:
            self.charList.insert(1, '[UNK]')
        if not self.charList:
            raise Exception('No characters found in character list')
        if use_mask:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList),
                num_oov_indices=num_oov_indices, mask_token='',
                oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                num_oov_indices=0, oov_token='', mask_token='',
                encoding="UTF-8", invert=True
            )
        else:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList),
                num_oov_indices=num_oov_indices, mask_token=None,
                oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                num_oov_indices=0, oov_token='', mask_token=None,
                encoding="UTF-8", invert=True
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
