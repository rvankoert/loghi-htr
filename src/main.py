from __future__ import division
from __future__ import print_function

import os
import json

from matplotlib import use
from word_beam_search import WordBeamSearch
from DataLoaderNew import DataLoaderNew
from utils import decode_batch_predictions
import re

import numpy as np
import random
import argparse
import editdistance
import subprocess
import matplotlib.pyplot as plt
from utils import Utils


def main():
    parser = argparse.ArgumentParser(
        description='Loghi HTR Core. Provides deep learning for Handwritten Text Recognition.')
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=str, default=-1,
                        help='gpu to be used, use -1 for CPU')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.0003,
                        help='learning_rate to be used, default 0.0003')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                        help='epochs to be used, default 40')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=4,
                        help='batch_size to be used, default 4')
    parser.add_argument('--num_workers', metavar='num_workers ', type=int, default=20,
                        help='num_workers')
    parser.add_argument('--max_queue_size', metavar='max_queue_size ', type=int, default=256,
                        help='max_queue_size')
    parser.add_argument('--height', metavar='height', type=int, default=32,
                        help='rescale everything to this height before training')
    parser.add_argument('--width', metavar='width', type=int, default=65536,
                        help='maximum width to be used. This should be a high number and generally does not need to '
                             'be changed')
    parser.add_argument('--channels', metavar='channels', type=int, default=3,
                        help='number of channels to use. 1 for grey-scale/binary images, three for color images, '
                             '4 for png\'s with transparency')
    # parser.add_argument('--loss', metavar='loss ', type=str, default="contrastive_loss",
    #                     help='contrastive_loss, binary_crossentropy, mse')
    # parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
    #                     help='optimizer: adam, adadelta, rmsprop, sgd')
    parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=0,
                        help='deprecated: memory_limit for gpu in MB. Default 0 for unlimited, in general keep this 0')

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
    # parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
    #                     help='testset to be used')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                        help='continue training/validation/testing/inferencing from this model as a starting point.')
    parser.add_argument('--model_name', metavar='model_name ', type=str, default=None,
                        help='use model_name in the output')
    parser.add_argument('--use_mask', help='whether or not to mask certain parts of the data', action='store_true')
    parser.add_argument('--use_gru', help='use GRU Gated Recurrent Units instead of LSTM in the recurrent layers',
                        action='store_true')
    parser.add_argument('--results_file', metavar='results_file', type=str, default='output/results.txt',
                        help='results_file. When inferencing the results are stored at this location.')
    parser.add_argument('--config_file_output', metavar='config_file_output', type=str, default=None,
                        help='config_file_output')
    parser.add_argument('--config_file', metavar='config_file', type=str, default='config.txt',
                        help='config_file')
    parser.add_argument('--greedy', help='use greedy ctc decoding. beam_width will be ignored', action='store_true')
    parser.add_argument('--beam_width', metavar='beam_width ', type=int, default=10,
                        help='beam_width when validating/inferencing, higher beam_width gets better results, but run '
                             'slower. Default 10')
    parser.add_argument('--decay_steps', metavar='decay_steps', type=int, default=-1,
                        help='decay_steps. default -1. After this number of iterations the learning rate will '
                             'decrease with 10 percent. When 0, it will not decrease. When -1 it is set to num_batches')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='beta: decay_rate. Default 0.99. disables learning rate decay when set to 0')

    parser.add_argument('--steps_per_epoch', metavar='steps_per_epoch ', type=int, default=None,
                        help='steps_per_epoch. default None')
    parser.add_argument('--model', metavar='model ', type=str, default=None,
                        help='Model to use')
    parser.add_argument('--batch_normalization', help='batch_normalization', action='store_true')
    parser.add_argument('--charlist', metavar='charlist ', type=str, default=None,
                        help='Charlist to use')
    parser.add_argument('--output_charlist', metavar='output_charlist', type=str, default=None,
                        help='output_charlist to use')
    parser.add_argument('--use_dropout', help='if enabled some dropout will be added to the model if creating a new '
                                              'model', action='store_true')
    parser.add_argument('--use_rnn_dropout', help='if enabled some dropout will be added to rnn layers of the model '
                                                  'if creating a new model', action='store_true')
    parser.add_argument('--rnn_layers', metavar='rnn_layers ', type=int, default=5,
                        help='number of rnn layers to use in the recurrent part. default 5')
    parser.add_argument('--rnn_units', metavar='rnn_units ', type=int, default=256,
                        help='numbers of units in each rnn_layer. default 256')
    parser.add_argument('--do_binarize_otsu', action='store_true',
                        help='beta: do_binarize_otsu')
    parser.add_argument('--do_binarize_sauvola', action='store_true',
                        help='beta: do_binarize_sauvola')
    parser.add_argument('--multiply', metavar='multiply ', type=int, default=1,
                        help='multiply training data, default 1')
    parser.add_argument('--replace_final_layer', action='store_true',
                        help='beta: replace_final_layer. You can do this to extend/decrease the character set when '
                             'using an existing model')
    parser.add_argument('--replace_recurrent_layer', action='store_true',
                        help='beta: replace_recurrent_layer. Set new recurrent layer using an existing model. '
                             'Additionally replaces final layer as well.')
    parser.add_argument('--thaw', action='store_true',
                        help='beta: thaw. thaws conv layers, only usable with existing_model')
    parser.add_argument('--freeze_conv_layers', action='store_true',
                        help='beta: freeze_conv_layers. Freezes conv layers, only usable with existing_model')
    parser.add_argument('--freeze_recurrent_layers', action='store_true',
                        help='beta: freeze_recurrent_layers. Freezes recurrent layers, only usable with existing_model')
    parser.add_argument('--freeze_dense_layers', action='store_true',
                        help='beta: freeze_dense_layers. Freezes dense layers, only usable with existing_model')
    parser.add_argument('--num_oov_indices', metavar='num_oov_indices ', type=int, default=0,
                        help='num_oov_indices, default 0, set to 1 if unknown characters are in dataset, but not in '
                             'charlist. Use when you get the error "consider setting `num_oov_indices=1`"')
    parser.add_argument('--corpus_file', metavar='corpus_file ', type=str, default=None,
                        help='beta: corpus_file to use, enables WordBeamSearch')
    parser.add_argument('--wbs_smoothing', metavar='corpus_file ', type=float, default=0.1,
                        help='beta: smoothing to use when using word beam search')
    # Data augmentations
    parser.add_argument('--elastic_transform', action='store_true',
                        help='beta: elastic_transform, currently disabled')
    parser.add_argument('--random_crop', action='store_true',
                        help='beta: broken. random_crop')
    parser.add_argument('--random_width', action='store_true',
                        help='data augmentation option: random_width, stretches the textline horizontally to random width')
    parser.add_argument('--distort_jpeg', action='store_true',
                        help='beta: distort_jpeg')
    parser.add_argument('--augment', action='store_true',
                        help='beta: apply data augmentation to training set. In general this is a good idea')

    parser.add_argument('--dropout_rnn', type=float, default=0.5,
                        help='beta: dropout_rnn. Default 0.5. Only used when use_dropout_rnn is enabled')
    parser.add_argument('--dropout_recurrent_dropout', type=float, default=0,
                        help='beta: dropout_recurrent_dropout. Default 0. This is terribly slow on GPU as there is no support in cuDNN RNN ops')
    parser.add_argument('--reset_dropout', action='store_true',
                        help='beta: reset_dropout')
    parser.add_argument('--set_dropout', type=float, default=0.5,
                        help='beta: set_dropout')
    parser.add_argument('--dropout_dense', type=float, default=0.5,
                        help='beta: dropout_dense')
    parser.add_argument('--dropoutconv', type=float, default=0.0,
                        help='beta: set_dropout')
    parser.add_argument('--ignore_lines_unknown_character', action='store_true',
                        help='beta: ignore_lines_unknown_character. Ignores during training/validation lines that '
                             'contain characters that are not in charlist.')
    parser.add_argument('--check_missing_files', action='store_true',
                        help='beta: check_missing_files')
    parser.add_argument('--use_float32', action='store_true',
                        help='beta: use_float32')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='beta: early_stopping_patience')
    parser.add_argument('--normalize_text', action='store_true',
                        help='')
    parser.add_argument('--use_lmdb', action='store_true',
                        help='use lmdb to store images, this might be faster for more epochs')
    parser.add_argument('--reuse_old_lmdb_train', type=str, help='path of the folder of lmdb for training data')
    parser.add_argument('--reuse_old_lmdb_val', type=str, help='path of the folder of lmdb for validation data')
    parser.add_argument('--reuse_old_lmdb_test', type=str, help='path of the folder of lmdb for test data')
    parser.add_argument('--reuse_old_lmdb_inference', type=str, help='path of the folder of lmdb for inference data')
    parser.add_argument('--deterministic', action='store_true',
                        help='beta: deterministic mode (reproducible results')
    parser.add_argument('--output_checkpoints', action='store_true',
                        help='Continuously output checkpoints after each epoch. Default only best_val is saved')
    parser.add_argument('--no_auto', action='store_true',
                        help='No Auto disabled automatic "fixing" of certain parameters')
    parser.add_argument('--cnn_multiplier', type=int, default=4,
                        help='beta: cnn_multiplier')

    args = parser.parse_args()

    if args.deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # place from/imports here so os.environ["CUDA_VISIBLE_DEVICES"]  is set before TF loads
    from Model import Model, CERMetric, WERMetric, CTCLoss
    from keras.utils.generic_utils import get_custom_objects
    import tensorflow.keras as keras
    import tensorflow as tf
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    do_train = args.do_train
    use_mask = args.use_mask
    if not args.no_auto and args.train_list:
        print('do_train implied by providing a train_list')
        do_train = True
    if not args.no_auto and args.batch_size > 1:
        print('batch_size > 1, setting use_mask=True')
        use_mask = True

    if args.gpu != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')

        # if len(gpus) > 0 and args.memory_limit > 0:
        #     print('setting memory_limit: ' + str(args.memory_limit))
        #     tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        #         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])
    if not args.use_float32 and args.gpu != '-1':
        print("using mixed_float16")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("using float32")

    learning_rate = args.learning_rate

    if args.output and not os.path.exists(args.output):
        try:
            os.mkdir(args.output)
        except OSError as error:
            print(error)
            print('can not create output directory')

    char_list = None
    charlist_location = args.charlist
    if not charlist_location:
        charlist_location = args.output + '/charlist.txt'
    model_channels = args.channels
    model_height = args.height
    with strategy.scope():
        if args.existing_model:
            if not os.path.exists(args.existing_model):
                print('cannot find existing model on disk: ' + args.existing_model)
                exit(1)
            if not os.path.exists(charlist_location):
                print('cannot find charlist on disk: ' + charlist_location)
                exit(1)
            with open(charlist_location) as file:
                char_list = list(char for char in file.read())
            # char_list = sorted(list(char_list))
            print("using charlist")
            print("length charlist: " + str(len(char_list)))
            print(char_list)
            get_custom_objects().update({"CERMetric": CERMetric})
            get_custom_objects().update({"WERMetric": WERMetric})
            get_custom_objects().update({"CTCLoss": CTCLoss})

            model = keras.models.load_model(args.existing_model)
            if not args.replace_final_layer:
                model_channels = model.layers[0].input_shape[0][3]

            model_height = model.layers[0].input_shape[0][2]
            if args.height != model_height:
                print('input height differs from model channels. use --height ' + str(model_height))
                print('resetting height to: ' + str(model_height))
                if args.no_auto:
                    exit(1)
            with open(args.charlist) as file:
                char_list = list(char for char in file.read())
        img_size = (model_height, args.width, model_channels)
        loader = DataLoaderNew(args.batch_size, img_size,
                               train_list=args.train_list,
                               validation_list=args.validation_list,
                               test_list=args.test_list,
                               inference_list=args.inference_list,
                               char_list=char_list,
                               do_binarize_sauvola=args.do_binarize_sauvola,
                               do_binarize_otsu=args.do_binarize_otsu,
                               multiply=args.multiply,
                               augment=args.augment,
                               elastic_transform=args.elastic_transform,
                               random_crop=args.random_crop,
                               random_width=args.random_width,
                               check_missing_files=args.check_missing_files,
                               distort_jpeg=args.distort_jpeg,
                               replace_final_layer=args.replace_final_layer,
                               normalize_text=args.normalize_text,
                               use_lmdb=args.use_lmdb,
                               reuse_old_lmdb_train=args.reuse_old_lmdb_train,
                               reuse_old_lmdb_val=args.reuse_old_lmdb_val,
                               reuse_old_lmdb_test=args.reuse_old_lmdb_test,
                               reuse_old_lmdb_inference=args.reuse_old_lmdb_inference,
                               use_mask=use_mask
                               )

        print("creating generators")
        training_generator, validation_generator, test_generator, inference_generator, utils, train_batches = loader.generators()

        # Testing
        if False:
            for run in range(1):
                print("testing dataloader " + str(run))
                training_generator.set_charlist(char_list, True, num_oov_indices=args.num_oov_indices)

                no_batches = training_generator.__len__()
                for i in range(10):
                    # if i%10 == 0:
                    print(i)
                    item = training_generator.__getitem__(i)
                training_generator.random_width = True
                training_generator.random_crop = True
                # training_generator.augment = True
                training_generator.elastic_transform = True
                training_generator.distort_jpeg = True
                training_generator.do_binarize_sauvola = False
                training_generator.do_binarize_otsu = False
                training_generator.on_epoch_end()
                for i in range(10):
                    # if i%10 == 0:
                    print(i)
                    batch = training_generator.__getitem__(i)
                    item = tf.image.convert_image_dtype(-0.5 - batch[0][1], dtype=tf.uint8)
                    gtImageEncoded = tf.image.encode_png(item)
                    tf.io.write_file("/tmp/test-" + str(i) + ".png", gtImageEncoded)

            exit()
        modelClass = Model()
        print(len(loader.charList))

        if args.decay_rate > 0 and args.decay_steps > 0:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=args.decay_steps,
                decay_rate=args.decay_rate)
        elif args.decay_rate > 0 and args.decay_steps == -1 and do_train:
            if training_generator is None:
                print('training, but training_generator is None. Did you provide a train_list?')
                exit(1)
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=train_batches,
                decay_rate=args.decay_rate)
        else:
            lr_schedule = learning_rate

        output_charlist_location = args.output_charlist
        if not output_charlist_location:
            output_charlist_location = args.output + '/charlist.txt'

        if args.existing_model:
            print('using existing model as base: ' + args.existing_model)
            # if not args.replace_final_layer:

            if args.replace_recurrent_layer:
                model = modelClass.replace_recurrent_layer(model, len(char_list), use_mask=use_mask,
                                                           use_gru=args.use_gru,
                                                           rnn_layers=args.rnn_layers, rnn_units=args.rnn_units,
                                                           use_rnn_dropout=args.use_rnn_dropout,
                                                           dropout_rnn=args.dropout_rnn)

            if args.replace_final_layer:
                with open(output_charlist_location, 'w') as chars_file:
                    chars_file.write(str().join(loader.charList))
                char_list = loader.charList

                model = modelClass.replace_final_layer(model, len(char_list), model.name, use_mask=use_mask)
            if args.thaw:
                for layer in model.layers:
                    layer.trainable = True

            if args.freeze_conv_layers:
                for layer in model.layers:
                    if layer.name.startswith("Conv"):
                        print(layer.name)
                        layer.trainable = False
            if args.freeze_recurrent_layers:
                for layer in model.layers:
                    if layer.name.startswith("bidirectional_"):
                        print(layer.name)
                        layer.trainable = False
            if args.freeze_dense_layers:
                for layer in model.layers:
                    if layer.name.startswith("dense"):
                        print(layer.name)
                        layer.trainable = False
            if args.reset_dropout:
                modelClass.set_dropout(model, args.set_dropout)

            # if True:
            #     for layer in model.layers:
            #         print(layer.name)
            #         layer.trainable = True

            model.compile(
                keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])
        else:
            # save characters of model for inference mode
            with open(output_charlist_location, 'w') as chars_file:
                chars_file.write(str().join(loader.charList))

            char_list = loader.charList
            print("creating new model")
            if 'new2' == args.model:
                model = modelClass.build_model_new2(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru)  # (loader.charList, keep_prob=0.8)
            elif 'new3' == args.model:
                model = modelClass.build_model_new3(img_size, len(char_list))  # (loader.charList, keep_prob=0.8)
            elif 'new4' == args.model:
                model = modelClass.build_model_new4(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    batch_normalization=args.batch_normalization)
            elif 'new5' == args.model:
                model = modelClass.build_model_new5(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    rnn_layers=5,
                                                    batch_normalization=args.batch_normalization,
                                                    dropout=args.use_dropout)
            elif 'new6' == args.model:
                model = modelClass.build_model_new6(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    rnn_layers=2,
                                                    batch_normalization=args.batch_normalization)
            elif 'new7' == args.model:
                model = modelClass.build_model_new7(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    rnn_layers=args.rnn_layers,
                                                    batch_normalization=args.batch_normalization,
                                                    dropout=args.use_dropout)
            elif 'new8' == args.model:
                model = modelClass.build_model_new8(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    rnn_layers=args.rnn_layers,
                                                    batch_normalization=args.batch_normalization,
                                                    dropout=args.use_dropout,
                                                    use_rnn_dropout=args.use_rnn_dropout)
            elif 'new9' == args.model:
                model = modelClass.build_model_new9(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru,
                                                    rnn_units=args.rnn_units,
                                                    rnn_layers=args.rnn_layers,
                                                    batch_normalization=args.batch_normalization,
                                                    dropout=args.use_dropout,
                                                    use_rnn_dropout=args.use_rnn_dropout)
            elif 'new10' == args.model:
                model = modelClass.build_model_new10(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     dropout=args.use_dropout,
                                                     use_rnn_dropout=args.use_rnn_dropout)
            elif 'new11' == args.model:
                model = modelClass.build_model_new11(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     dropout=args.use_dropout,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropoutconv=args.dropoutconv)
            elif 'new12' == args.model:
                model = modelClass.build_model_new12(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     dropout=args.use_dropout,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropoutconv=args.dropoutconv)
            elif 'new13' == args.model:
                model = modelClass.build_model_new13(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     dropout=args.use_dropout,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropoutconv=args.dropoutconv)
            elif 'new14' == args.model:
                model = modelClass.build_model_new14(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     dropout=args.use_dropout,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropout_recurrent_dropout=args.dropout_recurrent_dropout,
                                                     dropout_conv=args.dropoutconv,
                                                     dropout_dense=args.dropout_dense)
            elif 'new15' == args.model:
                model = modelClass.build_model_new15(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropout_recurrent_dropout=args.dropout_recurrent_dropout,
                                                     dropout_conv=args.dropoutconv,
                                                     dropout_dense=args.dropout_dense,
                                                     multiplier=args.cnn_multiplier)
            elif 'new16' == args.model:
                model = modelClass.build_model_new16(img_size, len(char_list),
                                                     use_mask=use_mask,
                                                     use_gru=args.use_gru,
                                                     rnn_units=args.rnn_units,
                                                     rnn_layers=args.rnn_layers,
                                                     batch_normalization=args.batch_normalization,
                                                     use_rnn_dropout=args.use_rnn_dropout,
                                                     dropout_rnn=args.dropout_rnn,
                                                     dropout_recurrent_dropout=args.dropout_recurrent_dropout,
                                                     dropout_conv=args.dropoutconv,
                                                     dropout_dense=args.dropout_dense,
                                                     multiplier=args.cnn_multiplier)
            elif 'old6' == args.model:
                model = modelClass.build_model_old6(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru)  # (loader.charList, keep_prob=0.8)
            elif 'old5' == args.model:
                model = modelClass.build_model_old5(img_size, len(char_list),
                                                    use_mask=use_mask,
                                                    use_gru=args.use_gru)  # (loader.charList, keep_prob=0.8)
            else:
                print(
                    'no model supplied. Existing or new ... Are you sure this is correct? use --model MODEL_HERE or --existing_model MODEL_HERE')
                exit()

            model.compile(
                keras.optimizers.Adam(learning_rate=lr_schedule), loss=CTCLoss, metrics=[CERMetric(), WERMetric()])


    model.summary(line_length=110)

    model_channels = model.layers[0].input_shape[0][3]
    if args.channels != model_channels:
        print('input channels differs from model channels. use --channels ' + str(model_channels))
        if args.no_auto:
            exit()
        else:
            print('setting correct number of channels: ' + str(model_channels))


    model_outputs = model.layers[-1].output_shape[2]
    num_characters = len(char_list) + 1
    if use_mask:
        num_characters = num_characters + 1
    if model_outputs != num_characters:
        print('model_outputs: ' + str(model_outputs))
        print('charlist: ' + str(num_characters))
        print('number of characters in model is different from charlist provided.')
        print('please find correct charlist and use --charlist CORRECT_CHARLIST')
        print('if the charlist is just 1 lower: did you forget --use_mask')
        exit(1)

    # test_generator = test_generator.getGenerator()
    # inference_dataset = inference_generator.getGenerator()

    if do_train:
        validation_dataset = None
        if args.validation_list:
            validation_dataset = validation_generator

        store_info(args, model)

        training_dataset = training_generator
        print('batches ' + str(training_dataset.__len__()))
        # try:
        history = Model().train_batch(
            model,
            training_dataset,
            validation_dataset,
            epochs=args.epochs,
            output=args.output,
            model_name='encoder12',
            steps_per_epoch=args.steps_per_epoch,
            num_workers=args.num_workers,
            max_queue_size=args.max_queue_size,
            early_stopping_patience=args.early_stopping_patience,
            output_checkpoints=args.output_checkpoints
        )

        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        if args.validation_list:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/CER")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output, 'plot.png'))

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["CER_metric"], label="CER train")
        if args.validation_list:
            plt.plot(history.history["val_CER_metric"], label="CER val")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/CER")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output, 'plot2.png'))

        # except tf.python.framework.errors_impl.ResourceExhaustedError as e:
        #     print("test")
    if args.do_validate:
        print("do_validate")
        utils = Utils(char_list, use_mask)
        validation_dataset = validation_generator
        # Get the prediction model by extracting layers till the output layer
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary(line_length=110)

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

        wbs = None
        if args.corpus_file:
            if not os.path.exists(args.corpus_file):
                print('cannot find corpus_file on disk: ' + args.corpus_file)
                exit(1)

            with open(args.corpus_file) as f:
                # # corpus = f.read()
                corpus = ''
                # # chars = set()
                for line in f:
                    # chars = chars.union(set(char for label in line for char in label))
                    if args.normalize_text:
                        line = loader.normalize(line)
                    corpus += line
            word_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÉØßàáâäçèéêëìïòóôõöøüōƒ̄ꞵ='
            word_chars = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzßàáâçèéëïñôöûüň'
            #
            chars = '' + ''.join(sorted(list(char_list)))

            print('using corpus file: ' + str(args.corpus_file))
            # NGramsForecast
            # Words
            # NGrams
            # NGramsForecastAndSample
            wbs = WordBeamSearch(args.beam_width, 'NGrams', args.wbs_smoothing, corpus.encode('utf8'), chars.encode('utf8'),
                                 word_chars.encode('utf8'))
            print('Created WordBeamSearcher')

        batch_no = 0
        #  Let's check results on some validation samples
        for batch in validation_dataset:
            # batch_images = batch["image"]
            # batch_labels = batch["label"]
            predictions = prediction_model.predict(batch[0])
            # preds = prediction_model.predict_on_batch(batch[0])
            predicted_texts = decode_batch_predictions(predictions, utils, args.greedy, args.beam_width,
                                                       args.num_oov_indices)

            # preds = utils.softmax(preds)
            predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
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
            if wbs:
                print('computing wbs...')
                label_str = wbs.compute(predsbeam)
                char_str = []  # decoded texts for batch
                # print(len(label_str))
                # print(label_str)
                for curr_label_str in label_str:
                    # print(len(curr_label_str))
                    s = ''.join([chars[label] for label in curr_label_str])
                    char_str.append(s)
                    print(s)

            counter += 1
            orig_texts = []
            for label in batch[1]:
                label = tf.strings.reduce_join(utils.num_to_char(label)).numpy().decode("utf-8")
                orig_texts.append(label.strip())

            for prediction in predicted_texts:
                for i in range(len(prediction)):
                    confidence = prediction[i][0]
                    predicted_text = prediction[i][1]
                    # for i in range(16):
                    original_text = orig_texts[i].strip().replace('', '')
                    predicted_text = predicted_text.strip().replace('', '')
                    current_editdistance = editdistance.eval(original_text, predicted_text)
                    current_editdistance_lower = editdistance.eval(original_text.lower(), predicted_text.lower())

                    pattern = re.compile('[\W_]+')
                    ground_truth_simple = pattern.sub('', original_text).lower()
                    predicted_simple = pattern.sub('', predicted_text).lower()
                    current_editdistance_simple = editdistance.eval(ground_truth_simple, predicted_simple)
                    if wbs:
                        predicted_wbs_simple = pattern.sub('', char_str[i]).lower()
                        current_editdistance_wbs_simple = editdistance.eval(ground_truth_simple, predicted_wbs_simple)
                        current_editdistance_wbs = editdistance.eval(original_text, char_str[i].strip())
                        current_editdistance_wbslower = editdistance.eval(original_text.lower(),
                                                                          char_str[i].strip().lower())
                    cer = current_editdistance / float(len(original_text))
                    if cer > 0.0:
                        # filename = validation_dataset.get_file(batch_no * batchSize + i)
                        # print(filename)
                        # print(predicted_simple)
                        print(original_text)
                        print(predicted_text)
                        if wbs:
                            print(char_str[i])
                    print('confidence: ' + str(confidence)
                          + ' cer: ' + str(cer)
                          + ' total_orig: ' + str(len(original_text))
                          + ' total_pred: ' + str(len(predicted_text))
                          + ' errors: ' + str(current_editdistance))
                    totaleditdistance += current_editdistance
                    totaleditdistance_lower += current_editdistance_lower
                    totaleditdistance_simple += current_editdistance_simple
                    if wbs:
                        totaleditdistance_wbs_simple += current_editdistance_wbs_simple
                        totaleditdistance_wbs += current_editdistance_wbs
                        totaleditdistance_wbs_lower += current_editdistance_wbslower
                    totallength += len(original_text)
                    totallength_simple += len(ground_truth_simple)

                    print(cer)
                    print("avg editdistance: " + str(totaleditdistance / float(totallength)))
                    print("avg editdistance lower: " + str(totaleditdistance_lower / float(totallength)))
                    if totallength_simple > 0:
                        print("avg editdistance simple: " + str(totaleditdistance_simple / float(totallength_simple)))
                    if wbs:
                        print("avg editdistance wbs: " + str(totaleditdistance_wbs / float(totallength)))
                        print("avg editdistance wbs lower: " + str(totaleditdistance_wbs_lower / float(totallength)))
                        if totallength_simple > 0:
                            print("avg editdistance wbs_simple: " + str(
                                totaleditdistance_wbs_simple / float(totallength_simple)))
            batch_no += 1

        totalcer = totaleditdistance / float(totallength)
        totalcerlower = totaleditdistance_lower / float(totallength)
        totalcersimple = totaleditdistance_simple / float(totallength_simple)
        if wbs:
            totalcerwbssimple = totaleditdistance_wbs_simple / float(totallength_simple)
            totalcerwbs = totaleditdistance_wbs / float(totallength)
            totalcerwbslower = totaleditdistance_wbs_lower / float(totallength)
        print('totalcer: ' + str(totalcer))
        print('totalcerlower: ' + str(totalcerlower))
        print('totalcersimple: ' + str(totalcersimple))
        if wbs:
            print('totalcerwbs: ' + str(totalcerwbs))
            print('totalcerwbslower: ' + str(totalcerwbslower))
            print('totalcerwbssimple: ' + str(totalcerwbssimple))
    #            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
    #            img = img.T
    #            title = f"Prediction: {pred_texts[i].strip()}"
    #            ax.imshow(img, cmap="gray")
    #            ax.set_title(title)
    #            ax.axis("off")
    #            plt.show()

    if args.do_inference:
        print('inferencing')
        print(char_list)
        loader = DataLoaderNew(args.batch_size,
                               img_size,
                               char_list,
                               inference_list=args.inference_list,
                               check_missing_files=args.check_missing_files,
                               normalize_text=args.normalize_text,
                               use_mask=use_mask
                               )
        training_generator, validation_generator, test_generator, inference_generator, utils, train_batches = loader.generators()
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense3").output
        )
        prediction_model.summary(line_length=110)

        inference_dataset = inference_generator

        store_info(args, model)

        batch_counter = 0
        with open(args.results_file, "w") as text_file:

            for batch in inference_dataset:
                predictions = prediction_model.predict_on_batch(batch[0])
                predicted_texts = decode_batch_predictions(predictions, utils, args.greedy, args.beam_width)

                orig_texts = []
                for label in batch[1]:
                    label = tf.strings.reduce_join(utils.num_to_char(label)).numpy().decode("utf-8")
                    orig_texts.append(label.strip())
                for prediction in predicted_texts:
                    for i in range(len(prediction)):
                        confidence = prediction[i][0]
                        predicted_text = prediction[i][1]
                        filename = loader.get_item('inference', (batch_counter * args.batch_size) + i)
                        original_text = orig_texts[i].strip().replace('', '')
                        predicted_text = predicted_text.strip().replace('', '')
                        print(original_text)
                        print(filename + "\t" + str(confidence) + "\t" + predicted_text)
                        text_file.write(filename + "\t" + str(confidence) + "\t" + predicted_text + "\n")

                    batch_counter += 1
                    text_file.flush()


def store_info(args, model):
    if os.path.exists("version_info"):
        with open("version_info") as file:
            version_info = file.read()
    else:
        bash_command = 'git log --format="%H" -n 1'
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, errors = process.communicate()
        version_info = output.decode('utf8', errors='strict').strip().replace('"', '')

    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))

    config = {
        'git_hash': version_info,
        'args': args.__dict__,
        'model': model_layers,
        'notes': ' '
    }

    if args.config_file_output:
        config_file_output = args.config_file_output
    else:
        config_file_output = os.path.join(args.output, 'config.json')
    with open(config_file_output, 'w') as configuration_file:
        json.dump(config, configuration_file)


if __name__ == '__main__':
    main()
