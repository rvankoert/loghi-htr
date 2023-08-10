import unittest
import sys
import tempfile
import os


class DataLoaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Determine the directory of this file
        current_file_dir = os.path.dirname(os.path.realpath(__file__))

        # Assuming the tests directory is at the root of your project,
        # get the project root
        project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
        src_dir = os.path.join(project_root, "src")
        sys.path.append(src_dir)

        # Set paths for data and model directories
        cls.data_dir = os.path.join(project_root, "tests", "data")

        cls.sample_image_paths = [os.path.join(
            cls.data_dir, f"test-image{i+1}") for i in range(3)]

        # Extract labels from .txt files
        cls.sample_labels = []
        for i in range(3):
            label_path = os.path.join(cls.data_dir, f"test-image{i+1}.txt")
            with open(label_path, 'r') as file:
                cls.sample_labels.append(file.readline().strip())

        # Create sample list file
        cls.sample_list_file = os.path.join(cls.data_dir, "sample_list.txt")
        with open(cls.sample_list_file, 'w') as f:
            for img_path, label in zip(cls.sample_image_paths,
                                       cls.sample_labels):
                f.write(f"{img_path}.png\t{label}\n")

        from DataLoaderNew import DataLoaderNew
        cls.DataLoader = DataLoaderNew

    def _create_temp_file(self, additional_lines=None):
        temp_sample_list_file = tempfile.NamedTemporaryFile(
            delete=False, mode='w+')

        for img_path, label in zip(self.sample_image_paths,
                                   self.sample_labels):
            temp_sample_list_file.write(f"{img_path}.png\t{label}\n")
        if additional_lines:
            for line in additional_lines:
                temp_sample_list_file.write(line + "\n")

        temp_sample_list_file.close()
        return temp_sample_list_file.name

    def _remove_temp_file(self, filename):
        os.remove(filename)

    def test_initialization(self):
        # Only provide the required arguments for initialization and check the
        # default values of the optional arguments
        batch_size = 32
        img_size = (256, 256, 3)

        data_loader = self.DataLoader(batch_size=batch_size,
                                      img_size=img_size)
        self.assertIsInstance(data_loader, self.DataLoader)

        # Check default values
        self.assertEqual(data_loader.batchSize, batch_size)
        self.assertEqual(data_loader.imgSize, img_size)

    def test_create_data_simple(self):
        # Sample data
        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        data_file_list = self.sample_list_file
        partition_name = 'test_partition'

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        # Call create_data
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, data_file_list)

        # Asserts
        self.assertEqual(len(files), 3)
        for i, (fileName, gtText) in enumerate(files):
            self.assertEqual(fileName, self.sample_image_paths[i] + '.png')
            self.assertEqual(gtText, self.sample_labels[i])

    def test_missing_files(self):
        # Manipulate sample file to have a missing image path
        additional_lines = [
            f"{os.path.join(self.data_dir, 'missing-image.png')}"
            "\tmissing_label"]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        # Sample data
        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        partition_name = 'test_partition'

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        # Call create_data with include_missing_files=False (default)
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # should still be 3, not 4, because we skip the missing file
        self.assertEqual(len(files), 3)

        # Call create_data with include_missing_files=True
        chars, files = data_loader.create_data(
            chars, labels, partition,
            partition_name, temp_sample_list_file, include_missing_files=True)

        # Asserts
        # should be 4 now, including the missing file
        self.assertEqual(len(files), 4)

    def test_unsupported_chars(self):
        # Sample data with unsupported characters
        additional_lines = [
            f"{self.sample_image_paths[0]}.png\tlabelX",
            f"{self.sample_image_paths[1]}.png\tlabelY"
        ]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        partition_name = 'test_partition'

        # Initialize DataLoader with injected_charlist set to a list without
        # 'X' and 'Y'
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))
        data_loader.injected_charlist = set(
            'abcdefghijklkmnopqrstuvwxyzN0123456789, ')\
            - set('XY')

        # Call create_data with include_unsupported_chars=False (default)
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # should still be 3, not 5, because we skip lines with 'X' and 'Y'
        self.assertEqual(len(files), 3)

        # Call create_data with include_unsupported_chars=True
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            temp_sample_list_file, include_unsupported_chars=True)

        # Asserts
        # should be 5 now, including the lines with 'X' and 'Y'
        self.assertEqual(len(files), 5)

        self._remove_temp_file(temp_sample_list_file)

    def _test_inference_mode(self):
        temp_sample_list_file = self._create_temp_file()

        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        partition_name = 'test_partition'

        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            temp_sample_list_file, is_inference=True)
        self.assertEqual(len(files), 3)
        for _, gtText in files:
            self.assertEqual(gtText, 'to be determined')

        self._remove_temp_file(temp_sample_list_file)

    def test_text_normalization(self):
        # Sample data with mixed-case labels
        additional_lines = [f"{self.sample_image_paths[0]}.png\tLabel ."]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        partition_name = 'test_partition'

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))
        # assuming normalization converts text to lowercase
        data_loader.normalize_text = True

        # Call create_data
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # last file's label should be normalized to 'label'
        self.assertEqual(files[-1][1], 'Label.')

        self._remove_temp_file(temp_sample_list_file)

    def test_multiplication(self):
        chars = set()
        labels = {'test_partition': []}
        partition = {'test_partition': []}
        data_file_list = self.sample_list_file
        partition_name = 'test_partition'

        # Initialize DataLoader with multiply set to 2
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))
        data_loader.multiply = 2

        # Call create_data with use_multiply=True
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            data_file_list, use_multiply=True)

        # Asserts
        # should be 6 now, as each line is duplicated due to multiplication
        self.assertEqual(len(files), 6)

    # TODO: tests for generators()
    #       tests for truncate_label()


if __name__ == "__main__":
    unittest.main()
