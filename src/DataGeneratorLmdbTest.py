import unittest

from DataGeneratorLmdb import DataGeneratorLmdb
from DataGeneratorNew import DataGeneratorNew


class DataGeneratorLmdbTest(unittest.TestCase):

    def test__datageneration_worksLmdb(self):
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

        lmdb.__getitem__(1)

    def test__datageneration_worksNew(self):
        labels = ['lem in de korte veerstraat, ons vervoegt', 'wonende meede binnen deze Stad in de g',
                          'verzogt betaling van de vorenstaande', '„zelve binnen een maand voldoen„',
                          'gaf, ik ben nu niet bij kas maar zal de']
        img_paths = ['test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l2.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l4.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l8.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l11.png',
                          'test/files/testset/NL-HlmNHA_1617_1604_0384/NL-HlmNHA_1617_1604_0384.xml-r1l10.png']

        charList = list(' &,-./123456789:=ABCDEFGHIJKMNOPRSTVWZabcdefghijklmnopqrstuvwxyz¶ê—„')

        generator = DataGeneratorNew(list_IDs=img_paths, labels=labels, charList=charList)

        generator.__getitem__(1)
