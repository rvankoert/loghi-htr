# Imports

# > Standard library
import argparse
import logging
import os
from typing import List

# > Third-party dependencies
import tensorflow as tf
from word_beam_search import WordBeamSearch

# > Local imports
from data.loader import DataLoader
from setup.config import Config
from utils.text import preprocess_text


def setup_word_beam_search(config: Config, charlist: List[str],
                           loader: DataLoader) -> WordBeamSearch:
    """
    Sets up the Word Beam Search (WBS) algorithm for use in character
    recognition tasks.

    Parameters
    ----------
    config : Config
        A Config object containing arguments related to the WBS setup, such as
        the path to the corpus file, beam width, and smoothing parameters.
    charlist : List[str]
        A list of characters used in the model.
    loader : DataLoader
        A data loader object capable of normalizing text using a specified
        normalization file.

    Returns
    -------
    WordBeamSearch
        An initialized WordBeamSearch object ready for use in decoding
        predictions.

    Raises
    ------
    FileNotFoundError
        If the corpus file specified in the arguments does not exist.

    Notes
    -----
    This function initializes a WordBeamSearch object with the given
    parameters. It loads a corpus file for language modeling and sets up the
    character sets for the WBS algorithm.
    """

    logging.info("Setting up WordBeamSearch...")

    args = config.args

    # Check if the corpus file exists
    if not os.path.exists(args.corpus_file):
        raise FileNotFoundError(f'Corpus file not found: {args.corpus_file}')

    # Load the corpus
    with open(args.corpus_file) as f:
        # Create the corpus
        corpus = ''
        for line in f:
            if args.normalization_file:
                line = loader.normalize(line, args.normalization_file)
            corpus += line
    logging.info(f'Using corpus file: {args.corpus_file}')

    # Create the WordBeamSearch object
    word_chars = \
        '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzßàáâçèéëïñôöûüň'
    chars = '' + ''.join(sorted(charlist))
    wbs = WordBeamSearch(args.beam_width, 'NGrams', args.wbs_smoothing,
                         corpus.encode('utf8'), chars.encode('utf8'),
                         word_chars.encode('utf8'))

    logging.info('Created WordBeamSearch')

    return wbs


def handle_wbs_results(predsbeam: tf.Tensor, wbs: WordBeamSearch,
                       chars: List[str]) -> List[str]:
    """
    Decodes batch predictions using Word Beam Search (WBS).

    Parameters
    ----------
    predsbeam : tf.Tensor
        The transposed predictions from the model, formatted for WBS
        processing.
    wbs : WordBeamSearch
        The WordBeamSearch object used for decoding.
    chars : List[str]
        A list of characters corresponding to the indices in the model's
        predictions.

    Returns
    -------
    List[str]
        A list of decoded strings from the batch predictions using Word Beam
        Search.

    Notes
    -----
    This function decodes each set of predictions in the batch using WBS and
    processes them to form readable strings. The function also handles any
    necessary preprocessing of the decoded text.
    """

    label_str = wbs.compute(predsbeam)
    char_str = []  # decoded texts for batch
    for curr_label_str in label_str:
        s = ''.join([chars[label-1] for label in curr_label_str])
        s = preprocess_text(s)
        char_str.append(s)

    return char_str
