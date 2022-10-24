from __future__ import division
from __future__ import print_function

import os
import random

from DataGeneratorNew import DataGeneratorNew


class DataLoaderNew:
    DTYPE = 'float32'

    currIdx = 0
    charList = []
    samples = []
    validation_dataset = []

    @staticmethod
    def normalize(input):
        output = input.replace(',,', '„') \
            .replace(' ,', ',') \
            .replace(',', ', ') \
            .replace(' .', '. ') \
            .replace('  ', ' ') \
            .replace('`', '\'') \
            .replace('´', '\'') \
            .replace('ʼ', '\'') \
            .replace('‘', '\'') \
            .replace('’', '\'') \
            .replace('“', '"') \
            .replace('”', '"') \
            .replace('·', '.') \
            .strip()
        return output

    def __init__(self, batchSize, imgSize, char_list=None,
                 train_list='',
                 validation_list='',
                 test_list='',
                 inference_list='',
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 normalize_text=True,
                 multiply=1,
                 augment=True,
                 elastic_transform=False,
                 num_oov_indices=0,
                 random_crop=False,
                 random_width=False,
                 check_missing_files=True,
                 distort_jpeg=False,
                 replace_final_layer=False
                 ):
        """loader for dataset at given location, preprocess images and text according to parameters"""

        # assert filePath[-1] == '/'

        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.height = imgSize[0]
        self.width = imgSize[1]
        self.channels = imgSize[2]
        self.partition = []
        self.injected_charlist = char_list
        self.train_list = train_list
        self.validation_list = validation_list
        self.test_list = test_list
        self.inference_list = inference_list
        self.do_binarize_sauvola = do_binarize_sauvola
        self.do_binarize_otsu = do_binarize_otsu
        self.normalize_text = normalize_text
        self.multiply = multiply
        self.dataAugmentation = augment
        self.elastic_transform = elastic_transform
        self.num_oov_indices = num_oov_indices
        self.random_crop = random_crop
        self.random_width = random_width
        self.check_missing_files = check_missing_files
        self.distort_jpeg = distort_jpeg
        self.replace_final_layer = replace_final_layer

    def generators(self):
        chars = set()
        partition = {'train': [], 'validation': [], 'test': [], 'inference': []}
        labels = {'train': [], 'validation': [], 'test': [], 'inference': []}
        trainLabels = {}
        valLabels = {}
        testLabels = {}
        inference_labels = {}
        print(self.train_list)

        if self.train_list:
            for sublist in self.train_list.split():
                if not os.path.exists(sublist):
                    print(sublist + "does not exist, enter a valid filename. exiting...")
                    exit(1)
                f = open(sublist)
                counter = 0
                for line in f:
                    # ignore comment line
                    if not line or line[0] == '#':
                        continue
                    # print(line)
                    lineSplit = line.strip().split('\t')
                    if len(lineSplit) == 1:
                        continue

                    # filename
                    fileName = lineSplit[0]
                    if self.check_missing_files and not os.path.exists(fileName):
                        print("missing: " + fileName)
                        continue
                    if self.normalize_text:
                        gtText = self.normalize(lineSplit[1])
                    else:
                        gtText = lineSplit[1]
                    ignoreLine = False
                    if self.injected_charlist and not self.replace_final_layer:
                        for char in gtText:
                            if char not in self.injected_charlist:
                                # print(self.injected_charlist)
                                print('a ignoring line: ' + gtText)
                                ignoreLine = True
                                break
                    if ignoreLine or not gtText:
                        continue
                    counter = counter + 1
                    for i in range(0, self.multiply):
                        partition['train'].append(fileName)
                        labels['train'].append(gtText)
                        trainLabels[fileName] = gtText
                    if not self.injected_charlist or self.replace_final_layer:
                        chars = chars.union(set(char for label in gtText for char in label))
                    # if counter > 100:
                    #     break
                f.close()

        if self.validation_list:
            for sublist in self.validation_list.split():
                if not os.path.exists(sublist):
                    print(sublist + "does not exist, enter a valid filename. exiting...")
                    exit(1)
                f = open(sublist)
                counter = 0
                for line in f:
                    # ignore comment line
                    if not line or line[0] == '#':
                        continue

                    lineSplit = line.strip().split('\t')
                    if len(lineSplit) == 1:
                        continue

                    # filename
                    fileName = lineSplit[0]
                    if self.check_missing_files and not os.path.exists(fileName):
                        print("missing: " + fileName)
                        continue
                    if self.normalize_text:
                        gtText = self.normalize(lineSplit[1])
                    else:
                        gtText = lineSplit[1]

                    ignoreLine = False
                    if self.injected_charlist and not self.replace_final_layer:
                        for char in gtText:
                            if char not in self.injected_charlist:
                                # print(self.injected_charlist)
                                print('b ignoring line: ' + gtText)
                                ignoreLine = True
                                break
                    if ignoreLine or not gtText:
                        continue

                    counter = counter + 1
                    # if (counter > 10000):
                    #     break

                    # put sample into list
                    partition['validation'].append(fileName)
                    labels['validation'].append(gtText)
                    valLabels[fileName] = gtText
                    if not self.injected_charlist or self.replace_final_layer:
                        chars = chars.union(set(char for label in gtText for char in label))
                f.close()

        counter = 0

        if self.test_list:
            for sublist in self.test_list.split():
                if not os.path.exists(sublist):
                    print(sublist + "does not exist, enter a valid filename. exiting...")
                    exit(1)
                f = open(sublist)
                for line in f:
                    # ignore comment line
                    if not line or line[0] == '#':
                        continue

                    lineSplit = line.strip().split('\t')
                    if len(lineSplit) == 1:
                        continue

                    # filename
                    fileName = lineSplit[0]
                    if self.check_missing_files and not os.path.exists(fileName):
                        print("missing: " + fileName)
                        continue
                    # img = cv2.imread(fileName)
                    # height, width, channels = img.shape
                    # if height < 20 or width < 100 or width / height < 4:
                    #     print(fileName)
                    #     os.remove(fileName)
                    #     continue
                    if self.normalize_text:
                        gtText = self.normalize(lineSplit[1])
                    else:
                        gtText = lineSplit[1]

                    if not gtText:
                        continue

                    counter = counter + 1
                    # if (counter > 100):
                    #     break

                    # put sample into list
                    partition['test'].append(fileName)
                    labels['test'].append(gtText)
                    testLabels[fileName] = gtText
                    if not self.injected_charlist:
                        chars = chars.union(set(char for label in gtText for char in label))
                f.close()

        if self.inference_list:
            for sublist in self.inference_list.split():
                if not os.path.exists(sublist):
                    print(sublist + "does not exist, enter a valid filename. exiting...")
                    exit(1)
                f = open(sublist)
                for line in f:
                    # ignore comment line
                    if not line or line[0] == '#':
                        continue

                    lineSplit = line.strip().split('\t')
                    assert len(lineSplit) >= 1

                    # filename
                    fileName = lineSplit[0]
                    # if not os.path.exists(fileName):
                    #     # print(fileName)
                    #     continue
                    # img = cv2.imread(fileName)
                    # height, width, channels = img.shape
                    # if height < 20 or width < 100 or width / height < 4:
                    #     print(fileName)
                    #     os.remove(fileName)
                    #     continue
                    label = 'to be determined'

                    counter = counter + 1
                    # if (counter > 100):
                    #     break
                    # if counter % 10000 == 0:
                    #     print(counter)

                    # put sample into list
                    partition['inference'].append(fileName)
                    labels['inference'].append(label)
                    inference_labels[fileName] = label
                f.close()
                if len(partition['inference']) == 0:
                    print("no data to inference. Check your input-file")
                    exit(1)
        # list of all chars in dataset
        if self.injected_charlist and not self.replace_final_layer:
            print('using injected_charlist')
            self.charList = self.injected_charlist
        else:
            self.charList = sorted(list(chars))

        trainParams = {'shuffle': True,
                       'batch_size': self.batchSize,
                       'height': self.height,
                       'channels': self.channels,
                       'do_binarize_sauvola': self.do_binarize_sauvola,
                       'do_binarize_otsu': self.do_binarize_otsu,
                       'augment': self.dataAugmentation,
                       'elastic_transform': self.elastic_transform,
                       'random_crop': self.random_crop,
                       'random_width': self.random_width,
                       'distort_jpeg': self.distort_jpeg
                       }
        validationParams = {'shuffle': False,
                            'batch_size': self.batchSize,
                            'height': self.height,
                            'channels': self.channels,
                            'do_binarize_sauvola': self.do_binarize_sauvola,
                            'do_binarize_otsu': self.do_binarize_otsu,
                            }
        testParams = {'shuffle': False,
                      'batch_size': self.batchSize,
                      'height': self.height,
                      'channels': self.channels,
                      'do_binarize_sauvola': self.do_binarize_sauvola,
                      'do_binarize_otsu': self.do_binarize_otsu,
                      }
        inference_params = {'shuffle': False,
                            'batch_size': self.batchSize,
                            'height': self.height,
                            'channels': self.channels,
                            'do_binarize_sauvola': self.do_binarize_sauvola,
                            'do_binarize_otsu': self.do_binarize_otsu,
                            }
        training_generator = None
        validation_generator = None
        test_generator = None
        inference_generator = None
        if self.train_list:
            training_generator = DataGeneratorNew(
                partition['train'],
                labels['train'],
                **trainParams,
                charList=self.charList,
                num_oov_indices=self.num_oov_indices
            )
        if self.validation_list:
            validation_generator = DataGeneratorNew(partition['validation'], labels['validation'], **validationParams,
                                                    charList=self.charList, num_oov_indices=self.num_oov_indices)
        if self.test_list:
            test_generator = DataGeneratorNew(partition['test'], labels['test'], **testParams, charList=self.charList,
                                              num_oov_indices=self.num_oov_indices)
        if self.inference_list:
            inference_generator = DataGeneratorNew(partition['inference'], labels['inference'], **inference_params,
                                                   charList=self.charList, num_oov_indices=self.num_oov_indices)

        self.partition = partition

        return training_generator, validation_generator, test_generator, inference_generator

    @staticmethod
    def truncateLabel(text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def get_item(self, partition, item_id):
        # print(self.partition)
        return self.partition[partition][item_id]
