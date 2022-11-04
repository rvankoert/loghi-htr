import unittest

from DataLoaderNew import DataLoaderNew


class DataLoaderNewTest(unittest.TestCase):

    def test_load_training_data(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list.txt'
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_training_data_ignore_missing_images(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list_with_missing_images.txt'
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_training_data_includes_missing_images_when_check_missing_files_is_false(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list_with_missing_images.txt',
            check_missing_files=False
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de', 'boven wel expresselyk hebben geprotesteer'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l13.png'])

    def test_load_training_data_ignores_lines_without_text(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list_with_textless_images.txt'
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png'])

    def test_load_training_data_ignores_lines_with_unsupported_characters(self):
        self.char_list = [' ', '"', '#', '%', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                          'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '§',
                          '½', 'à', 'â', 'ä', 'é', 'ë', 'ï', 'ƒ', '‛', '„', '‸', '⁓', '℔', '⅓']
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list_with_unsupported_characters.txt',
            char_list=self.char_list
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png'])

    def test_load_training_data_includes_lines_with_unsupported_characters_when_replace_final_layer_is_true(self):
        self.char_list = [' ', '"', '#', '%', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                          'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '§',
                          '½', 'à', 'â', 'ä', 'é', 'ë', 'ï', 'ƒ', '‛', '„', '‸', '⁓', '℔', '⅓']
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            train_list='./test/files/images_list_with_unsupported_characters.txt',
            char_list=self.char_list,
            replace_final_layer=True
        ).generators()
        self.assertEqual(training_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'ᚢnsᚢppᛟrted chars'])
        self.assertEqual(training_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_val_data(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            validation_list='./test/files/images_list.txt'
        ).generators()
        self.assertEqual(validation_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(validation_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_val_data_ignore_missing_images(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            validation_list='./test/files/images_list_with_missing_images.txt'
        ).generators()
        self.assertEqual(validation_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(validation_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_val_data_includes_missing_images_when_check_missing_files_is_false(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            validation_list='./test/files/images_list_with_missing_images.txt',
            check_missing_files=False
        ).generators()
        self.assertEqual(validation_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de', 'boven wel expresselyk hebben geprotesteer'])
        self.assertEqual(validation_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l13.png'])

    def test_load_val_data_ignores_lines_without_text(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            validation_list='./test/files/images_list_with_textless_images.txt'
        ).generators()
        self.assertEqual(validation_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„'])
        self.assertEqual(validation_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png'])

    def test_load_val_data_ignores_lines_with_unsupported_characters(self):
        self.char_list = [' ', '"', '#', '%', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                          'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '§',
                          '½', 'à', 'â', 'ä', 'é', 'ë', 'ï', 'ƒ', '‛', '„', '‸', '⁓', '℔', '⅓']
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            validation_list='./test/files/images_list_with_unsupported_characters.txt',
            char_list=self.char_list
        ).generators()
        self.assertEqual(validation_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„'])
        self.assertEqual(validation_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png'])

    def test_load_test_data(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            test_list='./test/files/images_list.txt'
        ).generators()
        self.assertEqual(test_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(test_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_test_data_ignore_missing_images(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            test_list='./test/files/images_list_with_missing_images.txt'
        ).generators()
        self.assertEqual(test_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de'])
        self.assertEqual(test_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_test_data_includes_missing_images_when_check_missing_files_is_false(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            test_list='./test/files/images_list_with_missing_images.txt',
            check_missing_files=False
        ).generators()
        self.assertEqual(test_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de', 'boven wel expresselyk hebben geprotesteer'])
        self.assertEqual(test_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l13.png'])

    def test_load_test_data_ignores_lines_without_text(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            test_list='./test/files/images_list_with_textless_images.txt'
        ).generators()
        self.assertEqual(test_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„'])
        self.assertEqual(test_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png'])

    def test_load_test_data_includes_lines_with_unsupported_characters(self):
        self.char_list = [' ', '"', '#', '%', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                          '8', '9', ':', ';', '=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                          'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '§',
                          '½', 'à', 'â', 'ä', 'é', 'ë', 'ï', 'ƒ', '‛', '„', '‸', '⁓', '℔', '⅓']
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            test_list='./test/files/images_list_with_unsupported_characters.txt',
            char_list=self.char_list
        ).generators()
        self.assertEqual(test_generator.labels,
                         ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'ᚢnsᚢppᛟrted chars'])
        self.assertEqual(test_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_inference_data(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            inference_list='./test/files/inference_list.txt'
        ).generators()

        self.assertEqual(inference_generator.labels,
                         ['to be determined', 'to be determined', 'to be determined', 'to be determined',
                          'to be determined'])
        self.assertEqual(inference_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png'])

    def test_load_inference_data_includes_missing_images(self):
        training_generator, validation_generator, test_generator, inference_generator = DataLoaderNew(
            batchSize=1,
            imgSize=[50, 100, 3],
            inference_list='./test/files/inference_list_with_missing_images.txt'
        ).generators()
        self.assertEqual(inference_generator.labels,
                         ['to be determined', 'to be determined', 'to be determined', 'to be determined',
                          'to be determined', 'to be determined'])
        self.assertEqual(inference_generator.list_IDs,
                         ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l13.png'])
