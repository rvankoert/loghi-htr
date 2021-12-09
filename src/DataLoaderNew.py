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

    def __init__(self, filePath, batchSize, imgSize, maxTextLen, train_size, charlist=None):
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
        self.filePath = filePath
        # f = open('/scratch/train_data_htr/linestripsnew/all.txt')
        # f = open('/home/rutger/training_all2.txt')
        f = open(filePath)

        chars = set()
        bad_samples = []
        i = 0
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue
            # print(line)
            lineSplit = line.strip().split('\t')
            assert len(lineSplit) >= 1

            fileName = lineSplit[0]

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[1:]), maxTextLen)
            if not gtText:
                continue
            gtText = gtText.strip()
            if not gtText:
                continue
            # chars = chars.union(set(list(gtText)))
            chars = chars.union(set(char for label in gtText for char in label))
            # check if image is not empty
            if not os.path.exists(fileName):
                print(fileName)
                continue

            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                print("bad sample: " + lineSplit[0])
                continue
            # img = cv2.imread(fileName)
            # height, width, channels = img.shape
            # # print (width *(height/ 32))
            # if height < 32 or width < 32 or width /(height / 32) < 2* len(gtText):
            #     print(fileName)
            #     # os.remove(fileName)
            #     continue
            # # n = text_file.write(line)

            # put sample into list
            # gtText = gtText.ljust(maxTextLen, '€')
            self.samples.append((gtText, fileName))
            i = i+1
            if i % 2000 == 0:
                print(i)
                # break
        # text_file.close()
        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if len(bad_samples) > 0:
            print("Warning, damaged images found:", bad_samples)
        print("load textlines")

        # split into training and validation set: 95% - 5%
        random.seed(42)
        # random.shuffle(self.samples)

        # list of all chars in dataset
        if charlist is None:
            self.charList = sorted(list(chars))
        else:
            self.charList = charlist


    def generators(self):
        partition = {'train': [], 'validation': [], 'test': [], 'inference' : []}
        labels = {'train': [], 'validation': [], 'test': [], 'inference' : []}
        trainLabels = {}
        valLabels = {}
        testLabels = {}
        inference_labels = {}
        print(self.filePath)
        f = open(self.filePath)
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
                continue
            label = lineSplit[1]

            counter = counter + 1
            partition['train'].append(fileName)
            labels['train'].append(label)
            trainLabels[fileName] = label
            # if (counter > 1000):
            #     break

        f = open("training_all_republic_val.txt")
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
            label = lineSplit[1]

            counter = counter + 1
            # if (counter > 1000):
            #     break

            # put sample into list
            partition['validation'].append(fileName)
            labels['validation'].append(label)
            valLabels[fileName] = label

        counter = 0

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
            label = lineSplit[1]

            counter = counter + 1
            # if (counter > 100):
            #     break

            # put sample into list
            partition['test'].append(fileName)
            labels['test'].append(label)
            testLabels[fileName] = label

        f = open(self.filePath)
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