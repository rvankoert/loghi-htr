from __future__ import division
from __future__ import print_function

import os

from tensorflow_addons import layers

from Model import Model, CERMetric, WERMetric
from DataLoader import DataLoader
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import argparse
import editdistance


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'

    fnCharList = '../model/charList2.txt'
    fnAccuracy = '../model/accuracy2.txt'
    fnInfer = '../data2/test.png'
    fnCorpus = '../data2/corpus.txt'

    modelOutput = '../models/model-val-best'

def main():

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        pred = tf.dtypes.cast(pred, tf.float32)
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :maxTextLen
                  ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(loader.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=int, default=-1,
                        help='gpu to be used')
    parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                        help='percent_validation to be used')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.0001,
                        help='learning_rate to be used')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                        help='epochs to be used')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1')
    parser.add_argument('--height', metavar='height', type=int, default=32,
                        help='height to be used')
    parser.add_argument('--width', metavar='width', type=int, default=2048,
                        help='width to be used')
    parser.add_argument('--channels', metavar='channels', type=int, default=3,
                        help='channels to be used')
    parser.add_argument('--do_train', help='train', action='store_true')
    parser.add_argument('--do_validate', help='validate', action='store_true')
    parser.add_argument('--do_inference', help='inference', action='store_true')

    parser.add_argument('--output', metavar='output', type=str, default='output',
                        help='base output to be used')
    parser.add_argument('--trainset', metavar='trainset', type=str, default='/home/rutger/training_all_ijsberg.txt',
                        help='trainset to be used')
    parser.add_argument('--testset', metavar='testset', type=str, default='/data/cvl-database-1-1/test.txt',
                        help='testset to be used')
    parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                        help='testset to be used')
    parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                        help='spec')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                        help='existing_model')
    parser.add_argument('--model_name', metavar='model_name ', type=str, default='difornet10',
                        help='model_name')
    parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
                        help='contrastive_loss, binary_crossentropy, mse')
    parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                        help='optimizer: adam, adadelta, rmsprop, sgd')
    parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=4096,
                        help='memory_limit for gpu. Default 4096')
    parser.add_argument('--train_size', metavar='    train_size', type=float, default=0.99,
                        help='learning_rate to be used')

    args = parser.parse_args()

    print (args.existing_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    batchSize = args.batch_size
    imgSize = (args.width, args.height, args.channels)
    maxTextLen = 128
    epochs = args.epochs
    learning_rate = args.learning_rate
    # load training data, create TF model
    # print (args.trainset)
    # print (batchSize)
    # print (imgSize)
    # print (maxTextLen)
    loader = DataLoader(args.trainset, batchSize, imgSize, maxTextLen, args.train_size, args.channels)
    # open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

    # print(loader.charList)

    modelClass = Model()
    print(len(loader.charList))
    if not args.existing_model:
        # save characters of model for inference mode
        chars_file = open(FilePaths.fnCharList, 'w')
        chars_file.write(str().join(loader.charList))
        chars_file.close()
        print("creating new model")
        model = modelClass.build_model(imgSize, len(loader.charList), learning_rate)  # (loader.charList, keep_prob=0.8)
        model.compile(keras.optimizers.Adam(learning_rate=learning_rate))
    else:
        charlist = set(char for char in open(FilePaths.fnCharList).read())
        model = keras.models.load_model(args.existing_model)
        loader.set_charlist(charlist)

    model.summary()

    batch = loader.getTrainDataSet()
    validation_dataset = loader.getValidationDataSet()

    if (args.do_train):
        history = Model().train_batch(model, batch, validation_dataset, epochs=epochs, filepath=FilePaths.modelOutput)


    if (args.do_validate):

        # Get the prediction model by extracting layers till the output layer
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary()

        totalcer = 0.
        totaleditdistance = 0
        totallength = 0
        counter = 0
        #  Let's check results on some validation samples
        for batch in validation_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]

            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)
            counter += 1
            orig_texts = []
            for label in batch_labels:
                label = tf.strings.reduce_join(loader.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())

     #       _, ax = plt.subplots(1,1, figsize=(1024,32 ))
            for i in range(len(pred_texts)):
                # for i in range(16):
                original_text = orig_texts[i].strip().replace('€', '')
                predicted_text = pred_texts[i].strip()
                print(original_text)
                print(predicted_text)
                cer = editdistance.eval(original_text, predicted_text)/float(len(original_text))
                totaleditdistance += editdistance.eval(original_text, predicted_text)
                totallength += len(original_text)
                totalcer += cer
                print(cer)
        totalcer = totaleditdistance/float(totallength)
        print('totalcer: ' + str(totalcer))
    #            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
    #            img = img.T
    #            title = f"Prediction: {pred_texts[i].strip()}"
    #            ax.imshow(img, cmap="gray")
    #            ax.set_title(title)
    #            ax.axis("off")
    #            plt.show()


    if (args.do_inference):

        loader = DataLoader(args.trainset, batchSize, imgSize, maxTextLen, 1)
        charlist = set(char for char in open(FilePaths.fnCharList).read())
        model = keras.models.load_model(args.existing_model)
        loader.set_charlist(charlist)

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary()
        inference_dataset = loader.getTrainDataSet()
        #  Let's check results on some validation samples
        for batch in inference_dataset:
            batch_images = batch["image"]
            batch_labels = batch["label"]

            preds = prediction_model.predict(batch_images)
            pred_texts = decode_batch_predictions(preds)

            orig_texts = []
            for label in batch_labels:
                label = tf.strings.reduce_join(loader.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())

            for i in range(len(pred_texts)):
                original_text = orig_texts[i].strip().replace('€', '')
                predicted_text = pred_texts[i].strip().replace('€', '')
                print(original_text)
                print(predicted_text)


if __name__ == '__main__':
    main()
