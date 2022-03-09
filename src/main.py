from __future__ import division
from __future__ import print_function

import os

from keras.layers import Dense
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
from word_beam_search import WordBeamSearch

from DataLoaderNew import DataLoaderNew
from utils import decode_batch_predictions
import re, string

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
    # tf.compat.v1.enable_eager_execution()
    # print(tf.executing_eagerly())

    # A utility function to decode the output of the network


    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=int, default=-1,
                        help='gpu to be used, use -1 for CPU')
    parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                        help='percent_validation to be used')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.001,
                        help='learning_rate to be used, default 0.001')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                        help='epochs to be used, default 40')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1, default 1')
    parser.add_argument('--height', metavar='height', type=int, default=32,
                        help='rescale everything to this height before training')
    parser.add_argument('--width', metavar='width', type=int, default=65536,
                        help='width to be used')
    parser.add_argument('--channels', metavar='channels', type=int, default=3,
                        help='number of channels to use. 1 for grey-scale/binary images, three for color images, '
                             '4 for png\'s with transparency')
    parser.add_argument('--do_train', help='enable the training. Use this flag if you want to train.',
                        action='store_true')
    parser.add_argument('--do_validate', help='if enabled a separate validation run will be done', action='store_true')
    parser.add_argument('--do_inference', help='inference', action='store_true')

    parser.add_argument('--output', metavar='output', type=str, default='output',
                        help='base output to be used')
    parser.add_argument('--train_list', metavar='train_list', type=str, default=None,
                        help='use this file containing textline location+transcription for training. You can use '
                             'multiple input files quoted and space separated "training_file1.txt '
                             'training_file2.txt"to combine training sets.')
    parser.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                        help='use this file containing textline location+transcription for validation. You can use '
                             'multiple input files quoted and space separated "validation_file1.txt '
                             'validation_file2.txt"to combine validation sets.')
    parser.add_argument('--test_list', metavar='test_list', type=str, default=None,
                        help='use this file containing textline location+transcription for testing. You can use '
                             'multiple input files quoted and space separated "test_file1.txt test_file2.txt"to '
                             'combine testing sets.')
    parser.add_argument('--inference_list', metavar='inference_list', type=str, default=None,
                        help='use this file containing textline location+transcription for inferencing. You can use '
                             'multiple input files quoted and space separated "inference_file1.txt '
                             'inference_file2.txt"to combine inferencing sets.')
    parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                        help='testset to be used')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                        help='continue training/validation/testing/inferencing from this model as a starting point.')
    parser.add_argument('--model_name', metavar='model_name ', type=str, default=None,
                        help='use model_name in the output')
    parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
                        help='contrastive_loss, binary_crossentropy, mse')
    parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                        help='optimizer: adam, adadelta, rmsprop, sgd')
    parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=4096,
                        help='memory_limit for gpu. Default 4096')
    parser.add_argument('--use_mask', help='whether or not to mask certain parts of the data', action='store_true')
    parser.add_argument('--use_gru', help='use GRU Gated Recurrent Units instead of LSTM in the recurrent layers', action='store_true')
    parser.add_argument('--results_file', metavar='results_file', type=str, default='output/results.txt',
                        help='results_file')
    parser.add_argument('--config_file_output', metavar='config_file_output', type=str, default='output/config.txt',
                        help='config_file_output')
    parser.add_argument('--config_file', metavar='config_file', type=str, default='config.txt',
                        help='config_file')
    parser.add_argument('--greedy', help='use greedy ctc decoding. beam_width will be ignored', action='store_true')
    parser.add_argument('--beam_width', metavar='beam_width ', type=int, default=10,
                        help='beam_width when validating/inferencing, higher beam_width gets better results, but run slower. Default 10')
    parser.add_argument('--decay_steps', metavar='decay_steps ', type=int, default=10000,
                        help='decay_steps. default 10000. After this number of iterations the learning rate will decrease with 10 percent.')
    parser.add_argument('--steps_per_epoch', metavar='steps_per_epoch ', type=int, default=None,
                        help='steps_per_epoch. default None')
    parser.add_argument('--model', metavar='model ', type=str, default=None,
                        help='Model to use')
    parser.add_argument('--batch_normalization', help='batch_normalization', action='store_true')
    parser.add_argument('--charlist', metavar='charlist ', type=str, default='../model/charList2.txt',
                        help='Charlist to use')
    parser.add_argument('--use_dropout', help='if enabled some dropout will be added to the model if creating a new model', action='store_true')
    parser.add_argument('--use_rnn_dropout', help='if enabled some dropout will be added to rnn layers of the model if creating a new model', action='store_true')
    parser.add_argument('--rnn_layers', metavar='rnn_layers ', type=int, default=2,
                        help='number of rnn layers to use in the recurrent part. default 2')
    parser.add_argument('--rnn_units', metavar='rnn_units ', type=int, default=512,
                        help='numbers of units in each rnn_layer. default 512')
    parser.add_argument('--do_binarize_otsu', action='store_true',
                        help='beta: do_binarize_otsu')
    parser.add_argument('--do_binarize_sauvola', action='store_true',
                        help='beta: do_binarize_sauvola')
    parser.add_argument('--multiply', metavar='multiply ', type=int, default=1,
                        help='multiply training data, default 1')
    parser.add_argument('--augment', action='store_true',
                        help='beta: apply data augmentation to training set. In general this is a good idea')

    args = parser.parse_args()

    # if args.help:
    #     parser.print_usage()

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
    do_binarize_sauvola = False
    if args.do_binarize_sauvola:
        do_binarize_sauvola = True
    do_binarize_otsu = False
    if args.do_binarize_otsu:
        do_binarize_otsu=True
    augment = False
    if args.augment:
        augment = True
    # char_list = set(char for char in open(args.charlist).read())
    # char_list = sorted(list(char_list))
    # print("using charlist")
    # print("length charlist: " + str(len(char_list)))
    # print(char_list)
    char_list = None
    loader = DataLoaderNew(batchSize, imgSize,
                           train_list=args.train_list,
                           validation_list=args.validation_list,
                           test_list=args.test_list,
                           inference_list=args.inference_list,
                           char_list=char_list,
                           do_binarize_sauvola=do_binarize_sauvola,
                           do_binarize_otsu=do_binarize_otsu,
                           multiply=args.multiply,
                           augment=augment
                           )

    if args.model_name:
        FilePaths.modelOutput = args.output + "/" + args.model_name

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
        model.compile(
            keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
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
                                                rnn_units=args.rnn_units,
                                                batch_normalization=batch_normalization)
        elif 'new5' == args.model:
            model = modelClass.build_model_new5(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=args.rnn_units, rnn_layers=5,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout)
        elif 'new6' == args.model:
            model = modelClass.build_model_new6(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=args.rnn_units, rnn_layers=2,
                                                batch_normalization=batch_normalization)
        elif 'new7' == args.model:
            model = modelClass.build_model_new7(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=args.rnn_units, rnn_layers=args.rnn_layers,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout)
        elif 'new8' == args.model:
            model = modelClass.build_model_new8(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=args.rnn_units, rnn_layers=args.rnn_layers,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout,
                                                use_rnn_dropout=args.use_rnn_dropout)
        elif 'new9' == args.model:
            model = modelClass.build_model_new9(imgSize, len(char_list), use_mask=use_mask, use_gru=use_gru,
                                                rnn_units=args.rnn_units, rnn_layers=args.rnn_layers,
                                                batch_normalization=batch_normalization, dropout=args.use_dropout,
                                                use_rnn_dropout=args.use_rnn_dropout)
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

        model.compile(
            keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
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
        history = Model().train_batch(
            model,
            training_dataset,
            validation_dataset,
            epochs=epochs,
            output=args.output,
            MODEL_NAME='encoder12',
            steps_per_epoch=args.steps_per_epoch)

    if (args.do_validate):
        print("do_validate")
        validation_generator.set_charlist(char_list, use_mask)
        validation_dataset = validation_generator.getGenerator()
        # Get the prediction model by extracting layers till the output layer
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary()

        # weights = model.get_layer(name="dense3").get_weights()
        # prediction_model = keras.models.Model(
        #     model.get_layer(name="image").input, model.get_layer(name="bidirectional_3").output
        # )
        # # print(weights)
        # new_column = np.random.uniform(-0.5, 0.5, size=(512, 1))
        # weights[0] = np.append(weights[0], new_column, axis=1)
        # new_column = np.random.uniform(-0.5, 0.5, 1)[0]
        # weights[1] = np.append(weights[1], new_column)
        # dense3 = Dense(148, activation='softmax', weights=weights, name='dense3')(prediction_model.output)
        # # output = Dense(148, activation='softmax')(dense3)
        # prediction_model = keras.Model(inputs=prediction_model.inputs, outputs=dense3)
        # prediction_model.summary(line_length=120)


        totalcer = 0.
        totaleditdistance = 0
        totaleditdistance_lower = 0
        totaleditdistance_simple = 0
        totaleditdistance_wbs_simple = 0
        totaleditdistance_wbs = 0
        totaleditdistance_wbs_lower = 0
        totallength = 0
        totallength_simple = 0
        counter = 0

        corpus_file = "training_all_ijsberg_train_corpus.txt"
        f = open(corpus_file)
        # # corpus = f.read()
        corpus = ''
        # # chars = set()
        for line in f:
            # chars = chars.union(set(char for label in line for char in label))
            corpus += line
        # print (corpus)
        # corpus = 'a ba'  # two words "a" and "ba", separated by whitespace
        # chars = 'ab '  # the characters that can be recognized (in this order)
        # word_chars = corpus.splitlines()[0]  # characters that form words
        word_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÉØßàáâäçèéêëìïòóôõöøüōƒ̄ꞵ='
        #
        chars = '' + ''.join(sorted(list(char_list)))
        # chars = '' + ''.join(char_list)
        # print (word_chars)

        print(len(chars))
        # NGramsForecast
        # Words
        # NGrams
        # NGramsForecastAndSample
        wbs = WordBeamSearch(args.beam_width, 'NGrams', 0.9, corpus, chars,
                             word_chars)

        #  Let's check results on some validation samples
        for batch in validation_dataset:
            # batch_images = batch["image"]
            # batch_labels = batch["label"]
            preds = prediction_model.predict(batch[0])
            pred_texts = decode_batch_predictions(preds, maxTextLen, validation_generator, args.greedy, args.beam_width)
            predsbeam = tf.transpose(preds, perm=[1, 0, 2])

            # wbs = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
            #                      word_chars.encode('utf8'))
            # label_str = wbs.compute(mat)
            #
            # # result is string of labels terminated by blank
            # char_str = []
            # for curr_label_str in label_str:
            #     s = ''
            #     for label in curr_label_str:
            #         s += chars[label]  # map label to char
            #     char_str.append(s)
            # print(label_str[0])
            # print(char_str[0])
            # return label_str[0], char_str[0]


            # if True:
            #     exit(1)
            label_str = wbs.compute(predsbeam)
            # print('end label_str')

            # print(label_str)
            #
            # # compute label string

            # label_str = validation_generator.num_to_char(label_str)
            char_str = []  # decoded texts for batch
            # print(len(label_str))
            # print(label_str)
            for curr_label_str in label_str:
                # print(len(curr_label_str))
                s = ''.join([chars[label] for label in curr_label_str])
                char_str.append(s)
                # print(s)


            counter += 1
            orig_texts = []
            for label in batch[1]:
                label = tf.strings.reduce_join(validation_generator.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())

            for pred in pred_texts:
                for i in range(len(pred)):
                    confidence = pred[i][0]
                    pred_text = pred[i][1]
                    # for i in range(16):
                    original_text = orig_texts[i].strip().replace('', '')
                    predicted_text = pred_text.strip().replace('', '')
                    current_editdistance = editdistance.eval(original_text, predicted_text)
                    current_editdistance_lower = editdistance.eval(original_text.lower(), predicted_text.lower())

                    pattern = re.compile('[\W_]+')
                    ground_truth_simple = pattern.sub('', original_text).lower()
                    predicted_simple = pattern.sub('', predicted_text).lower()
                    predicted_wbs_simple = pattern.sub('', char_str[i]).lower()
                    current_editdistance_simple = editdistance.eval(ground_truth_simple, predicted_simple)
                    current_editdistance_wbs_simple = editdistance.eval(ground_truth_simple, predicted_wbs_simple)
                    current_editdistance_wbs = editdistance.eval(original_text, char_str[i].strip())
                    current_editdistance_wbslower = editdistance.eval(original_text.lower(), char_str[i].strip().lower())
                    cer = current_editdistance/float(len(original_text))
                    if cer >= 0.0:
                        print(predicted_simple)
                        print(original_text)
                        print(predicted_text)
                        print(char_str[i])

                    totaleditdistance += current_editdistance
                    totaleditdistance_lower += current_editdistance_lower
                    totaleditdistance_simple += current_editdistance_simple
                    totaleditdistance_wbs_simple += current_editdistance_wbs_simple
                    totaleditdistance_wbs += current_editdistance_wbs
                    totaleditdistance_wbs_lower += current_editdistance_wbslower
                    totallength += len(original_text)
                    totallength_simple += len(ground_truth_simple)

                    print(cer)
                    print(totaleditdistance/float(totallength))
                    print(totaleditdistance_lower / float(totallength))
                    print(totaleditdistance_simple/ float(totallength_simple))
                    print(totaleditdistance_wbs_simple/ float(totallength_simple))
                    print(totaleditdistance_wbs / float(totallength))
                    print(totaleditdistance_wbs_lower / float(totallength))
        totalcer = totaleditdistance/float(totallength)
        totalcerlower = totaleditdistance_lower/float(totallength)
        totalcersimple = totaleditdistance_simple / float(totallength_simple)
        totalcerwbssimple = totaleditdistance_wbs_simple / float(totallength_simple)
        totalcerwbs = totaleditdistance_wbs/float(totallength)
        totalcerwbslower = totaleditdistance_wbs_lower/float(totallength)
        print('totalcer: ' + str(totalcer))
        print('totalcerlower: ' + str(totalcerlower))
        print('totalcersimple: ' + str(totalcersimple))
        print('totalcerwbssimple: ' + str(totalcerwbssimple))
        print('totalcerwbs: ' + str(totalcerwbs))
        print('totalcerwbslower: ' + str(totalcerwbslower))
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
        loader = DataLoaderNew(batchSize, imgSize, char_list, inference_list=args.inference_list)
        training_generator, validation_generator, test_generator, inference_generator = loader.generators()
        inference_generator.set_charlist(char_list, use_mask=use_mask)
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary()
        inference_dataset = inference_generator.getGenerator()

        # write out used config:
        config_output_file = open(args.config_file_output, "w")
        config_output_file.write("seed="+str(args.seed) + "\n")
        config_output_file.write("gpu="+str(args.gpu) + "\n")
        config_output_file.write("percent_validation="+str(args.percent_validation) + "\n")
        config_output_file.write("learning_rate="+str(args.learning_rate) + "\n")
        config_output_file.write("epochs="+str(args.epochs) + "\n")
        config_output_file.write("batch_size="+str(args.batch_size) + "\n")
        config_output_file.write("height="+str(args.height) + "\n")
        config_output_file.write("width="+str(args.width) + "\n")
        config_output_file.write("channels="+str(args.channels) + "\n")
        config_output_file.write("output="+args.output + "\n")
        config_output_file.write("train_list="+args.train_list + "\n")
        config_output_file.write("validation_list="+args.validation_list + "\n")
        config_output_file.write("test_list="+args.test_list + "\n")
        config_output_file.write("inference_list="+args.inference_list + "\n")
        config_output_file.write("existing_model="+args.existing_model + "\n")
        config_output_file.write("model_name="+args.model_name + "\n")
        config_output_file.write("loss="+args.loss + "\n")
        config_output_file.write("optimizer="+args.optimizer + "\n")
        config_output_file.write("memory_limit="+str(args.memory_limit) + "\n")
        config_output_file.write("results_file="+args.results_file + "\n")
        config_output_file.write("config_file_output="+args.config_file_output + "\n")
        config_output_file.write("config_file="+args.config_file + "\n")
        config_output_file.write("beam_width="+str(args.beam_width) + "\n")
        config_output_file.write("decay_steps="+str(args.decay_steps) + "\n")
        config_output_file.write("steps_per_epoch="+str(args.steps_per_epoch) + "\n")
        config_output_file.write("model="+args.model + "\n")
        config_output_file.write("charlist="+args.charlist + "\n")
        config_output_file.write("inference_list="+args.inference_list + "\n")
        config_output_file.write("rnn_layers="+str(args.rnn_layers) + "\n")
        config_output_file.write("rnn_units="+str(args.rnn_units) + "\n")
        config_output_file.write("multiply="+str(args.multiply) + "\n")

        # parser.add_argument('--do_train', help='enable the training. Use this flag if you want to train.',
        #                     action='store_true')
        # parser.add_argument('--do_validate', help='if enabled a separate validation run will be done',
        #                     action='store_true')
        # parser.add_argument('--do_inference', help='inference', action='store_true')

        # parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
        #                     help='testset to be used')
        # parser.add_argument('--use_mask', help='whether or not to mask certain parts of the data', action='store_true')
        # parser.add_argument('--use_gru', help='use GRU Gated Recurrent Units instead of LSTM in the recurrent layers',
        #                     action='store_true')
        # parser.add_argument('--greedy', help='use greedy ctc decoding. beam_width will be ignored', action='store_true')
        # parser.add_argument('--batch_normalization', help='batch_normalization', action='store_true')
        # parser.add_argument('--use_dropout',
        #                     help='if enabled some dropout will be added to the model if creating a new model',
        #                     action='store_true')
        # parser.add_argument('--use_rnn_dropout',
        #                     help='if enabled some dropout will be added to rnn layers of the model if creating a new model',
        #                     action='store_true')
        # parser.add_argument('--do_binarize_otsu', action='store_true',
        #                     help='beta: do_binarize_otsu')
        # parser.add_argument('--do_binarize_sauvola', action='store_true',
        #                     help='beta: do_binarize_sauvola')
        # parser.add_argument('--augment', action='store_true',
        #                     help='beta: apply data augmentation to training set. In general this is a good idea')

        config_output_file.close()
        #  Let's check results on some validation samples
        batch_counter = 0
        text_file = open(args.results_file, "w")

        for batch in inference_dataset:
            # batch_images = batch["image"]
            # batch_labels = batch["label"]
            # prediction_model.reset_state()
            preds = prediction_model.predict_on_batch(batch[0])
            pred_texts = decode_batch_predictions(preds, maxTextLen, inference_generator, args.greedy, args.beam_width)

            orig_texts = []
            for label in batch[1]:
                label = tf.strings.reduce_join(inference_generator.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())
            for pred in pred_texts:
                for i in range(len(pred)):
                    confidence = pred[i][0]
                    pred_text = pred[i][1]
                    # for i in range(16):
                    filename = loader.get_item('inference', (batch_counter * batchSize) + i)
                    original_text = orig_texts[i].strip().replace('', '')
                    predicted_text = pred_text.strip().replace('', '')
                    print(original_text)
                    print(filename + "\t" + str(confidence) + "\t" + predicted_text)
                    text_file.write(filename + "\t" + str(confidence) + "\t" + predicted_text + "\n")

                    # for i in range(len(pred_texts)):
            #     filename = loader.get_item(batch_counter * batchSize + item_counter)
            #     original_text = orig_texts[i].strip().replace('', '')
            #     predicted_text = pred_texts[i].strip().replace('', '')
                batch_counter += 1
            # keras.backend.clear_session()
        text_file.close()

if __name__ == '__main__':
    main()
