from __future__ import division
from __future__ import print_function

import os

from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.framework import sparse_tensor, dtypes
from tensorflow.python.ops import array_ops, math_ops, sparse_ops
from tensorflow_addons import layers

from Model import Model, CERMetric, WERMetric, CTCLoss
# from DataLoader import DataLoader
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import argparse
import editdistance
# import warpctc_tensorflow
# from word_beam_search import WordBeamSearch

from DataLoaderNew import DataLoaderNew
from utils import decode_batch_predictions


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
    # tf.compat.v1.disable_eager_execution()



    # A utility function to decode the output of the network


    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=int, default=-1,
                        help='gpu to be used')
    parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                        help='percent_validation to be used')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.001,
                        help='learning_rate to be used')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=1,
                        help='epochs to be used')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1')
    parser.add_argument('--height', metavar='height', type=int, default=32,
                        help='height to be used')
    parser.add_argument('--width', metavar='width', type=int, default=65536,
                        help='width to be used')
    parser.add_argument('--channels', metavar='channels', type=int, default=3,
                        help='channels to be used')
    parser.add_argument('--do_train', help='train', action='store_true')
    parser.add_argument('--do_validate', help='validate', action='store_true')
    parser.add_argument('--do_inference', help='inference', action='store_true')

    parser.add_argument('--output', metavar='output', type=str, default='output',
                        help='base output to be used')
    parser.add_argument('--train_list', metavar='train_list', type=str, default=None,
                        help='train_list to be used')
    parser.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                        help='validation_list to be used')
    parser.add_argument('--test_list', metavar='test_list', type=str, default=None,
                        help='test_list to be used')
    parser.add_argument('--inference_list', metavar='inference_list', type=str, default=None,
                        help='inference_list to be used')
    parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                        help='testset to be used')
    parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                        help='spec')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                        help='existing_model')
    parser.add_argument('--model_name', metavar='model_name ', type=str, default=None,
                        help='model_name')
    parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
                        help='contrastive_loss, binary_crossentropy, mse')
    parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                        help='optimizer: adam, adadelta, rmsprop, sgd')
    parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=4096,
                        help='memory_limit for gpu. Default 4096')
    parser.add_argument('--train_size', metavar='train_size', type=float, default=0.99,
                        help='learning_rate to be used')
    parser.add_argument('--use_mask', help='use_mask', action='store_true')
    parser.add_argument('--use_gru', help='use_gru', action='store_true')
    parser.add_argument('--results_file', metavar='results_file', type=str, default='output/results.txt',
                        help='results_file')
    parser.add_argument('--greedy', help='use greedy ctc decoding. beam_width will be ignored', action='store_true')
    parser.add_argument('--beam_width', metavar='beam_width ', type=int, default=10,
                        help='beam_width. default 10')
    parser.add_argument('--decay_steps', metavar='decay_steps ', type=int, default=10000,
                        help='decay_steps. default 10000')
    parser.add_argument('--steps_per_epoch', metavar='steps_per_epoch ', type=int, default=None,
                        help='steps_per_epoch. default None')
    parser.add_argument('--model', metavar='model ', type=str, default=None,
                        help='Model to use')
    parser.add_argument('--batch_normalization', help='batch_normalization', action='store_true')
    parser.add_argument('--charlist', metavar='charlist ', type=str, default='../model/charList2.txt',
                        help='Charlist to use')
    parser.add_argument('--use_dropout', help='dropout', action='store_true')
    parser.add_argument('--rnn_layers', metavar='rnn_layers ', type=int, default=2,
                        help='rnn_layers. default 2')


    args = parser.parse_args()

    print(args.existing_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    batchSize = args.batch_size
    imgSize = (args.height, args.width, args.channels)
    maxTextLen = 128
    epochs = args.epochs
    learning_rate = args.learning_rate
    # load training data, create TF model
    # print (args.trainset)
    # print (batchSize)
    # print (imgSize)
    # print (maxTextLen)

    char_list = set(char for char in open(args.charlist).read())
    char_list = sorted(list(char_list))
    print("using charlist")
    print("length charlist: " + str(len(char_list)))
    print(char_list)
    char_list = None
    loader = DataLoaderNew(batchSize, imgSize, maxTextLen, args.train_size,
                           train_list=args.train_list,
                           validation_list=args.validation_list,
                           test_list=args.test_list,
                           inference_list=args.inference_list,
                           char_list=char_list
                           )

    if args.model_name:
        FilePaths.modelOutput = '../models/'+args.model_name

    print("creating generators")
    training_generator, validation_generator, test_generator, inference_generator = loader.generators()

    modelClass = Model()
    print(len(loader.charList))
    use_gru = False
    use_mask = False
    batch_normalization = False
    if args.use_gru:
        use_gru = True
    if args.use_mask:
        use_mask = True
    if args.batch_normalization:
        batch_normalization = True

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=args.decay_steps,
        decay_rate=0.90)

    if args.existing_model:
        char_list = list(char for char in open(args.charlist).read())
        # char_list = sorted(list(char_list))
        print("using charlist")
        print("length charlist: " + str(len(char_list)))
        print(char_list)
        get_custom_objects().update({"CERMetric": CERMetric})
        get_custom_objects().update({"WERMetric": WERMetric})
        get_custom_objects().update({"CTCLoss": CTCLoss})

        model = keras.models.load_model(args.existing_model)

        if True:
            for layer in model.layers:
                print(layer.name)
                layer.trainable = True


        # model.compile(keras.optimizers.Adam(learning_rate=lr_schedule), metrics=[CERMetric(), WERMetric()])
        # model.compile(keras.optimizers.Adam(learning_rate=lr_schedule))
        model.compile(keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
        # model.compile(keras.optimizers.SGD(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
    else:
        # save characters of model for inference mode
        chars_file = open(args.charlist, 'w')
        chars_file.write(str().join(loader.charList))
        chars_file.close()
        char_list = loader.charList
        print("creating new model")
        # model = modelClass.build_model(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        # model = modelClass.build_model_new2(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        if 'new2' == args.model:
            model = modelClass.build_model_new2(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        elif 'new3' == args.model:
            model = modelClass.build_model_new3(imgSize, len(char_list))  # (loader.charList, keep_prob=0.8)
        elif 'new4' == args.model:
            model = modelClass.build_model_new4(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=512,
                                                batch_normalization=batch_normalization)
        elif 'new5' == args.model:
            model = modelClass.build_model_new5(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=512, rnn_layers=5,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout)
        elif 'new6' == args.model:
            model = modelClass.build_model_new6(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=512, rnn_layers=2,
                                                batch_normalization=batch_normalization)
        elif 'new7' == args.model:
            model = modelClass.build_model_new7(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=512, rnn_layers=args.rnn_layers,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout)
        elif 'old6' == args.model:
            model = modelClass.build_model_old6(imgSize, len(char_list), use_mask=use_mask,
                                                use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        elif 'old5' == args.model:
            model = modelClass.build_model_old5(imgSize, len(char_list), use_mask=use_mask,
                                                use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        # Old models that require specific loader
        # elif 'old4' == args.model:
        #     model = modelClass.build_model_old4(imgSize, len(char_list), use_mask=use_mask,
        #                                         use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        # elif 'old3' == args.model:
        #     model = modelClass.build_model_old3(imgSize, len(char_list), use_mask=use_mask,
        #                                         use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        # elif 'old2' == args.model:
        #     model = modelClass.build_model_old2(imgSize, len(char_list), learning_rate=learning_rate)  # (loader.charList, keep_prob=0.8)
        # elif 'old1' == args.model:
        #     model = modelClass.build_model_old1(imgSize, len(char_list), learning_rate=learning_rate)  # (loader.charList, keep_prob=0.8)
        else:
            print('using default model ... Are you sure this is correct?')
            model = modelClass.build_model_new2(imgSize, len(char_list), use_mask=use_mask,
                                                use_gru=use_gru)  # (loader.charList, keep_prob=0.8)
        # model = modelClass.build_model_new1(
        #     imgSize,
        #     input_dim=maxTextLen + 1,
        #     output_dim=len(char_list),
        #     rnn_units=512,
        # )

        model.compile(keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
        # model.compile(keras.optimizers.RMSprop(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])

    model.summary(line_length=110)




    # test_generator = test_generator.getGenerator()
    # inference_dataset = inference_generator.getGenerator()

    if (args.do_train):
        validation_dataset = None
        if args.do_validate:
            validation_generator.set_charlist(char_list, use_mask)
            validation_dataset = validation_generator.getGenerator()
        training_generator.set_charlist(char_list, use_mask)
        training_dataset = training_generator.getGenerator()
        history = Model().train_batch(model, training_dataset, validation_dataset, epochs=epochs, filepath=FilePaths.modelOutput, MODEL_NAME='encoder12', steps_per_epoch=args.steps_per_epoch)

    if (args.do_validate):
        print("do_validate")
        validation_generator.set_charlist(char_list, use_mask)
        validation_dataset = validation_generator.getGenerator()
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
            # batch_images = batch["image"]
            # batch_labels = batch["label"]

            preds = prediction_model.predict(batch[0])
            pred_texts = decode_batch_predictions(preds, maxTextLen, validation_generator, args.greedy, args.beam_width)

            # corpus = 'a ba'  # two words "a" and "ba", separated by whitespace
            # chars = 'ab '  # the characters that can be recognized (in this order)
            # word_chars = 'ab'  # characters that form words
            #
            # wbs = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf-8'),
            #                      word_chars.encode('utf8'))
            #
            # # compute label string
            # label_str = wbs.compute(pred_texts[0])

            counter += 1
            orig_texts = []
            for label in batch[1]:
                label = tf.strings.reduce_join(validation_generator.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())

            for pred_text in pred_texts:
                for i in range(len(pred_text)):
                    # for i in range(16):
                    original_text = orig_texts[i].strip().replace('', '')
                    predicted_text = pred_text[i].strip().replace('', '')
                    print(original_text)
                    print(predicted_text)
                    current_editdistance = editdistance.eval(original_text, predicted_text)
                    cer = current_editdistance/float(len(original_text))
                    totaleditdistance += current_editdistance
                    totallength += len(original_text)
                    print(cer)
                    print(totaleditdistance/float(totallength))
        totalcer = totaleditdistance/float(totallength)
        print('totalcer: ' + str(totalcer))
    #            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
    #            img = img.T
    #            title = f"Prediction: {pred_texts[i].strip()}"
    #            ax.imshow(img, cmap="gray")
    #            ax.set_title(title)
    #            ax.axis("off")
    #            plt.show()


    if args.do_inference:
        print('inferencing')
        # char_list = set(char for char in open(args.charlist).read())
        # char_list = sorted(list(char_list))
        #
        print(char_list)
        loader = DataLoaderNew(batchSize, imgSize, maxTextLen, args.train_size, char_list, inference_list=args.inference_list)
        training_generator, validation_generator, test_generator, inference_generator = loader.generators()
        validation_generator.set_charlist(char_list, use_mask)
        inference_generator.set_charlist(char_list, use_mask=use_mask)
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary()
        inference_dataset = inference_generator.getGenerator()

        #  Let's check results on some validation samples
        batch_counter = 0
        text_file = open(args.results_file, "w")

        for batch in inference_dataset:
            # batch_images = batch["image"]
            # batch_labels = batch["label"]
            # prediction_model.reset_state()
            preds = prediction_model.predict_on_batch(batch[0])
            pred_texts = decode_batch_predictions(preds, maxTextLen, validation_generator, args.greedy, args.beam_width)

            orig_texts = []
            for label in batch[1]:
                label = tf.strings.reduce_join(inference_generator.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())
            for pred_text in pred_texts:
                item_counter = 0
                for i in range(len(pred_text)):
                    # for i in range(16):
                    filename = loader.get_item('inference', (batch_counter * batchSize) + item_counter)
                    original_text = orig_texts[i].strip().replace('', '')
                    predicted_text = pred_text[i].strip().replace('', '')
                    print(original_text)
                    print(filename + "\t" + predicted_text)
                    text_file.write(filename + "\t" + predicted_text + "\n")

                    # for i in range(len(pred_texts)):
            #     filename = loader.get_item(batch_counter * batchSize + item_counter)
            #     original_text = orig_texts[i].strip().replace('', '')
            #     predicted_text = pred_texts[i].strip().replace('', '')
                    item_counter += 1
                batch_counter += 1
            # keras.backend.clear_session()
        text_file.close()

if __name__ == '__main__':
    main()
