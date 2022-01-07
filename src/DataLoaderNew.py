from __future__ import division
from __future__ import print_function

import os
import random

from DataGenerator import DataGenerator


class DataLoaderNew:
    DTYPE = 'float32'

    dataAugmentation = False
    currIdx = 0
    charList = []
    samples = []
    validation_dataset = [];
    train_size =0.99

    def __init__(self, batchSize, imgSize, maxTextLen, train_size, char_list=None,
                 train_list='',
                 validation_list='',
                 test_list='',
                 inference_list=''):
        """loader for dataset at given location, preprocess images and text according to parameters"""

        # assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.train_size = train_size
        self.height = imgSize[0]
        self.width = imgSize[1]
        self.channels = imgSize[2]
        self.partition = []
        self.injected_charlist = char_list
        self.train_list = train_list
        self.validation_list = validation_list
        self.test_list = test_list
        self.inference_list = inference_list

    def generators(self):
        chars = set()
        partition = {'train': [], 'validation': [], 'test': [], 'inference' : []}
        labels = {'train': [], 'validation': [], 'test': [], 'inference' : []}
        trainLabels = {}
        valLabels = {}
        testLabels = {}
        inference_labels = {}
        print(self.train_list)

        if self.train_list:
            f = open(self.train_list)
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
                if not os.path.exists(fileName):
                    print(fileName)
                    continue
                gtText = lineSplit[1]

                counter = counter + 1
                partition['train'].append(fileName)
                labels['train'].append(gtText)
                trainLabels[fileName] = gtText
                chars = chars.union(set(char for label in gtText for char in label))
                # if (counter > 1000):
                #     break
            f.close()

        if self.validation_list:
            f = open(self.validation_list)
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
                if not os.path.exists(fileName):
                    # print(fileName)
                    continue
                gtText = lineSplit[1]

                counter = counter + 1
                # if (counter > 1000):
                #     break

                # put sample into list
                partition['validation'].append(fileName)
                labels['validation'].append(gtText)
                valLabels[fileName] = gtText
                chars = chars.union(set(char for label in gtText for char in label))
            f.close()

        counter = 0

        if self.test_list:
            f = open("training_all_republic_test.txt")
            for line in f:
                # ignore comment line
                if not line or line[0] == '#':
                    continue

                lineSplit = line.strip().split('\t')
                if len(lineSplit) == 1:
                    continue

                # filename
                fileName = lineSplit[0]
                if not os.path.exists(fileName):
                    # print(fileName)
                    continue
                # img = cv2.imread(fileName)
                # height, width, channels = img.shape
                # if height < 20 or width < 100 or width / height < 4:
                #     print(fileName)
                #     os.remove(fileName)
                #     continue
                gtText = lineSplit[1]

                counter = counter + 1
                # if (counter > 100):
                #     break

                # put sample into list
                partition['test'].append(fileName)
                labels['test'].append(gtText)
                testLabels[fileName] = gtText
                chars = chars.union(set(char for label in gtText for char in label))

        if self.inference_list:
            f = open(self.inference_list)
            for line in f:
                # ignore comment line
                if not line or line[0] == '#':
                    continue

                lineSplit = line.strip().split('\t')
                assert len(lineSplit) >= 1

                # filename
                fileName = lineSplit[0]
                if not os.path.exists(fileName):
                    # print(fileName)
                    continue
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

                # put sample into list
                partition['inference'].append(fileName)
                labels['inference'].append(label)
                inference_labels[fileName] = label

        # list of all chars in dataset
        if self.injected_charlist:
            self.charList = self.injected_charlist
        else:
            self.charList = sorted(list(chars))

        trainParams = {'shuffle': True,
                       'batch_size': self.batchSize,
                       'height': self.height,
                       'channels': self.channels
                       }
        validationParams = {'shuffle': False,
                            'batch_size': self.batchSize,
                            'height': self.height,
                            'channels': self.channels
                            }
        testParams = {'shuffle': False,
                      'batch_size': self.batchSize,
                      'height': self.height,
                      'channels': self.channels
                      }
        inference_params = {'shuffle': False,
                      'batch_size': self.batchSize,
                      'height': self.height,
                      'channels': self.channels
                      }
        training_generator = DataGenerator(partition['train'], labels['train'], **trainParams, charList=self.charList)
        validation_generator = DataGenerator(partition['validation'], labels['validation'], **validationParams,
                                             charList=self.charList)
        test_generator = DataGenerator(partition['test'], labels['test'], **testParams, charList=self.charList)
        inference_generator = DataGenerator(partition['inference'], labels['inference'], **inference_params, charList=self.charList)
        self.partition = partition

        return training_generator, validation_generator, test_generator, inference_generator


    def truncateLabel(self, text, maxTextLen):
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

    def get_item(self, item_id):
        # print(self.partition)
        return self.partition['inference'][item_id]

    def get_item(self, partition, item_id):
        # print(self.partition)
        return self.partition[partition][item_id]