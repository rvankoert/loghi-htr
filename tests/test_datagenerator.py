# Imports

# > Third party dependencies

# > Standard library
import logging
from pathlib import Path
import sys
import unittest


class TestDataGenerator(unittest.TestCase):
    """
    Tests for the data_generator class.

    Test coverage:
        1. `test_initialization` tests that the instance variables are
        initialized correctly.
    """

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

        from data.generator import DataGenerator
        cls.DataGenerator = DataGenerator

        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

    def test_initialization(self):
        tokenizer = self.Tokenizer(chars="ABC", use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, batch_size=32, height=128)

        # Verify that the instance variables are initialized correctly.
        self.assertEqual(dg.batch_size, 32)
        self.assertEqual(dg.height, 128)
        self.assertEqual(dg.channels, 1)


if __name__ == '__main__':
    unittest.main()
