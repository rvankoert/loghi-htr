# Imports

# > Third party dependencies
import tensorflow as tf
import numpy as np

# > Standard library
import logging
import unittest
from pathlib import Path
import sys


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        # Add the src directory to the path
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

        from utils.utils import Tokenizer
        cls.Tokenizer = Tokenizer

    def test_tokenizer_class(self):
        # Test without mask and no oov indices
        tokenizer = self.Tokenizer(chars=['a', 'b', 'c'], use_mask=False)
        self.assertEqual(tokenizer.charList, ['a', 'b', 'c'])

        # Test with mask
        tokenizer = self.Tokenizer(chars=['a', 'b', 'c'], use_mask=True)
        self.assertTrue(isinstance(tokenizer.char_to_num,
                        tf.keras.layers.StringLookup))
        self.assertTrue(tokenizer.char_to_num.mask_token, '')

        # Test set_charlist function with no oov indices.
        # Setting OOV indices to a value > 1 is broken.
        tokenizer = self.Tokenizer(chars=['a', 'b', 'c'],
                                   use_mask=False)
        tokenizer.set_charlist(chars=['a', 'b', 'c', 'd'],
                               use_mask=False, num_oov_indices=0)
        self.assertEqual(tokenizer.charList, ['a', 'b', 'c', 'd'])
        self.assertTrue(isinstance(tokenizer.char_to_num,
                        tf.keras.layers.StringLookup))

    def test_ctc_decode_greedy(self):
        # Mock data
        y_pred = np.random.random((32, 10, 5))
        input_length = np.random.randint(1, 10, size=(32,))

        # Call the function with greedy=True
        from utils.decoding import ctc_decode
        decoded_dense, log_prob = ctc_decode(y_pred, input_length,
                                             greedy=True)

        # Verify that the output is as expected
        self.assertTrue(isinstance(decoded_dense[0], tf.Tensor))
        self.assertTrue(isinstance(log_prob, tf.Tensor))

    def test_ctc_decode_beam(self):
        # Mock data
        y_pred = np.random.random((32, 10, 5))
        input_length = np.random.randint(1, 10, size=(32,))
        beam_width = 100

        # Call the function with greedy=False
        from utils.decoding import ctc_decode
        decoded_dense, log_prob = ctc_decode(y_pred, input_length,
                                             greedy=False,
                                             beam_width=beam_width)

        # Verify that the output is as expected
        # Ensure that the output is a list of tensors
        self.assertTrue(isinstance(decoded_dense, list))
        self.assertTrue(isinstance(decoded_dense[0], tf.Tensor))
        self.assertTrue(isinstance(log_prob, tf.Tensor))

    def test_decode_batch(self):
        chars = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(chars=chars, use_mask=False)

        # Mock data
        y_pred = np.random.random((32, 10, 5))

        # Call the function
        from utils.decoding import decode_batch_predictions
        result = decode_batch_predictions(y_pred, tokenizer)

        # Verify that the output is as expected
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], np.float32))
        self.assertTrue(isinstance(result[0][1], str))

    def test_decode_batch_with_beam(self):
        chars = ['a', 'b', 'c']
        tokenizer = self.Tokenizer(chars=chars, use_mask=False)

        # Mock data
        y_pred = np.random.random((32, 10, 5))

        # Call the function
        from utils.decoding import decode_batch_predictions
        result = decode_batch_predictions(y_pred, tokenizer, beam_width=100)

        # Verify that the output is as expected
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][0], np.float32))
        self.assertTrue(isinstance(result[0][1], str))


if __name__ == '__main__':
    unittest.main()
