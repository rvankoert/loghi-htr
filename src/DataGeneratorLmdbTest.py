import unittest
import numpy as np

from DataGeneratorLmdb import DataGeneratorLmdb
from DataGeneratorNew import DataGeneratorNew


class DataGeneratorLmdbTest(unittest.TestCase):

    def test_datageneration_works(self):
        labels = ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png']

        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList)

        self.assertEqual(5, len(lmdb))

    def test_datageneration_works_with_black_and_white(self):
        labels = ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png']

        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, channels=1)

        self.assertEqual(5, len(lmdb))

    def test_lmdb_image_equals_read_from_disk(self):
        labels = ['lem in de korte veerstraat, ons vervoegt']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png']
        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList)
        generator = DataGeneratorNew(list_IDs=img_paths, labels=labels, charList=charList)

        gen_images, gen_labels = generator[0]
        lmdb_images, lmdb_labels = lmdb[0]

        np.testing.assert_array_equal(gen_images.numpy(), lmdb_images.numpy())
        np.testing.assert_array_equal(gen_labels.numpy(), lmdb_labels.numpy())

    def test_lmdb_image_equals_read_from_disk_sauvola(self):
        labels = ['lem in de korte veerstraat, ons vervoegt']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png']
        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, do_binarize_sauvola=True, channels=1)
        generator = DataGeneratorNew(list_IDs=img_paths, labels=labels, charList=charList, do_binarize_sauvola=True, channels=1)

        gen_images, gen_labels = generator[0]
        lmdb_images, lmdb_labels = lmdb[0]

        np.testing.assert_array_equal(gen_images.numpy(), lmdb_images.numpy())
        np.testing.assert_array_equal(gen_labels.numpy(), lmdb_labels.numpy())

    def test_lmdb_image_is_difference_with_sauvola(self):
        labels = ['lem in de korte veerstraat, ons vervoegt']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png']
        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, do_binarize_sauvola=False)
        sauvola = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, do_binarize_sauvola=True, channels=1)

        (sauvola_images, sauvola_labels) = sauvola[0]
        (lmdb_images, lmdb_labels) = lmdb[0]

        np.testing.assert_array_equal(lmdb_labels.numpy(), sauvola_labels.numpy())
        self.assertTrue((lmdb_images.numpy() != sauvola_images.numpy()).any())

    def test_if_old_lmdb_is_reused(self):
        labels = ['lem in de korte veerstraat, ons vervoegt']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png']
        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        lmdb1 = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, shuffle=False)

        lmdb_name = lmdb1.lmdb_name
        lmdb2 = DataGeneratorLmdb(list_IDs=img_paths, labels=labels, charList=charList, reuse_old_lmdb=lmdb_name, shuffle=False)

        (lmdb1_images, lmdb1_labels) = lmdb1[0]
        (lmdb2_images, lmdb2_labels) = lmdb2[0]

        np.testing.assert_array_equal(lmdb1_labels.numpy(), lmdb2_labels.numpy())
        np.testing.assert_array_equal(lmdb1_images.numpy(), lmdb2_images.numpy())





