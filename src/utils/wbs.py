# Imports

# > Standard library
import logging
import os

# > Third-party dependencies
from word_beam_search import WordBeamSearch

# > Local imports
from utils.text import preprocess_text


def setup_word_beam_search(args, charlist, loader):
    logging.info("Setting up WordBeamSearch...")

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


def handle_wbs_results(predsbeam, wbs, args, chars):
    label_str = wbs.compute(predsbeam)
    char_str = []  # decoded texts for batch
    for curr_label_str in label_str:
        s = ''.join([chars[label-1] for label in curr_label_str])
        s = preprocess_text(s)
        char_str.append(s)

    return char_str
