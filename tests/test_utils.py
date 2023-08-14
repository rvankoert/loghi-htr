# Imports

# > Third party dependencies
import tensorflow as tf
import numpy as np

# > Standard library
import unittest
import os
import sys


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Determine the directory of this file
        current_file_dir = os.path.dirname(os.path.realpath(__file__))

        # Assuming the tests directory is at the root of your project,
        # get the project root
        project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
        src_dir = os.path.join(project_root, "src")
        sys.path.append(src_dir)

        from utils import Utils
        cls.utils = Utils

    def test_utils_class(self):
        # Test without mask and no oov indices
        utils = self.utils(chars=['a', 'b', 'c'], use_mask=False)
        self.assertEqual(utils.charList, ['a', 'b', 'c'])

        # Test with mask
        utils = self.utils(chars=['a', 'b', 'c'], use_mask=True)
        self.assertTrue(isinstance(utils.char_to_num,
                        tf.keras.layers.StringLookup))
        self.assertTrue(utils.char_to_num.mask_token, '')

        # Test set_charlist function with no oov indices.
        # Setting OOV indices to a value > 1 is broken.
        utils = self.utils(chars=['a', 'b', 'c'],
                           use_mask=False)
        utils.set_charlist(chars=['a', 'b', 'c', 'd'],
                           use_mask=False, num_oov_indices=0)
        self.assertEqual(utils.charList, ['a', 'b', 'c', 'd'])
        self.assertTrue(isinstance(utils.char_to_num,
                        tf.keras.layers.StringLookup))

    def test_ctc_decode_greedy(self):
        # Mock data
        y_pred = np.random.random((32, 10, 5))
        input_length = np.random.randint(1, 10, size=(32,))

        # Call the function with greedy=True
        from utils import ctc_decode
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
        top_paths = 1

        # Call the function with greedy=False
        from utils import ctc_decode
        decoded_dense, log_prob = ctc_decode(y_pred, input_length,
                                             greedy=False,
                                             beam_width=beam_width,
                                             top_paths=top_paths)

        # Verify that the output is as expected
        # Ensure that the output is a list of tensors
        self.assertTrue(isinstance(decoded_dense, list))
        self.assertTrue(isinstance(decoded_dense[0], tf.Tensor))
        self.assertTrue(isinstance(log_prob, tf.Tensor))

    def test_decode_batch(self):
        chars = ['a', 'b', 'c']
        utils = self.utils(chars=chars, use_mask=False)

        # Mock data
        y_pred = np.random.random((32, 10, 5))

        # Call the function
        from utils import decode_batch_predictions
        result = decode_batch_predictions(y_pred, utils)

        # Verify that the output is as expected
        self.assertTrue(isinstance(result[0], list))
        self.assertTrue(isinstance(result[0][0][0], np.float32))
        self.assertTrue(isinstance(result[0][0][1], str))

    def test_decode_batch_with_beam(self):
        chars = ['a', 'b', 'c']
        utils = self.utils(chars=chars, use_mask=False)

        # Mock data
        y_pred = np.random.random((32, 10, 5))

        # Call the function
        from utils import decode_batch_predictions
        result = decode_batch_predictions(y_pred, utils, beam_width=100)

        # Verify that the output is as expected
        self.assertTrue(isinstance(result[0], list))
        self.assertTrue(isinstance(result[0][0][0], np.float32))
        self.assertTrue(isinstance(result[0][0][1], str))


if __name__ == '__main__':
    unittest.main()
