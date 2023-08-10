import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Loghi HTR Core. Provides deep learning '
                    'for Handwritten Text Recognition.')

    # General args
    general_args = parser.add_argument_group('General arguments')
    general_args.add_argument('--gpu', metavar='gpu', type=str, default=-1,
                              help='gpu to be used, use -1 for CPU')
    general_args.add_argument('--output', metavar='output', type=str, default='output',
                              help='base output to be used')
    general_args.add_argument('--batch_size', metavar='batch_size', type=int, default=4,
                              help='batch_size to be used, default 4')
    general_args.add_argument('--results_file', metavar='results_file', type=str, default='output/results.txt',
                              help='results_file. When inferencing the results are stored at this location.')
    general_args.add_argument('--config_file_output', metavar='config_file_output', type=str, default=None,
                              help='config_file_output')
    general_args.add_argument('--optimizer', metavar='optimizer ', type=str, default='adam',
                              help='optimizer.')
    general_args.add_argument('--seed', metavar='seed', type=int, default=42,
                              help='random seed to be used')
    general_args.add_argument('--channels', metavar='channels', type=int, default=3,
                              help='number of channels to use. 1 for grey-scale/binary images, three for color images, '
                              '4 for png\'s with transparency')
    general_args.add_argument('--max_queue_size', metavar='max_queue_size ', type=int, default=256,
                              help='max_queue_size')
    general_args.add_argument(
        '--use_mask', help='whether or not to mask certain parts of the data. Defaults to true when batch_size > 1', action='store_true')
    general_args.add_argument('--charlist', metavar='charlist ', type=str, default=None,
                              help='Charlist to use')
    general_args.add_argument('--output_charlist', metavar='output_charlist', type=str, default=None,
                              help='output_charlist to use')

    # Training args
    training_args = parser.add_argument_group('General training arguments')
    training_args.add_argument('--do_train',
                               help='enable the training. '
                                    'Use this flag if you want to train.',
                               action='store_true')
    training_args.add_argument('--learning_rate', metavar='learning_rate',
                               type=float, default=0.0003,
                               help='learning_rate to be used, default 0.0003')
    training_args.add_argument('--epochs', metavar='epochs', type=int,
                               default=40, help='epochs to be used, default 40')
    training_args.add_argument('--height', metavar='height', type=int, default=64,
                               help='rescale everything to this height before '
                                    'training, default 64')
    training_args.add_argument('--width', metavar='width', type=int, default=65536,
                               help='maximum width to be used. '
                                    'This should be a high number and '
                                    'generally does not need to be changed')
    training_args.add_argument('--train_list', metavar='train_list',
                               type=str, default=None,
                               help='use this file containing textline '
                                    'location+transcription for training. '
                                    'You can use multiple input files '
                                    'quoted and space separated '
                                    '"training_file1.txt training_file2.txt" '
                                    'to combine training sets.')
    training_args.add_argument('--decay_steps', metavar='decay_steps', type=int, default=-1,
                               help='decay_steps. default -1. After this number of iterations the learning rate will '
                               'decrease with 10 percent. When 0, it will not decrease. When -1 it is set to num_batches / 1 epoch')
    training_args.add_argument('--decay_rate', type=float, default=0.99,
                               help='beta: decay_rate. Default 0.99. disables learning rate decay when set to 0')

    training_args.add_argument('--steps_per_epoch', metavar='steps_per_epoch ', type=int, default=None,
                               help='steps_per_epoch. default None')
    training_args.add_argument('--output_checkpoints', action='store_true',
                               help='Continuously output checkpoints after each epoch. Default only best_val is saved')
    training_args.add_argument('--use_float32', action='store_true',
                               help='beta: use_float32')
    training_args.add_argument('--early_stopping_patience', type=int, default=20,
                               help='beta: early_stopping_patience')
    training_args.add_argument('--multiply', metavar='multiply ', type=int, default=1,
                               help='multiply training data, default 1')
    training_args.add_argument(
        '--do_validate', help='if enabled a separate validation run will be done', action='store_true')

    training_args.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                               help='use this file containing textline location+transcription for validation. You can use '
                               'multiple input files quoted and space separated "validation_file1.txt '
                               'validation_file2.txt"to combine validation sets.')
    training_args.add_argument('--test_list', metavar='test_list', type=str, default=None,
                               help='use this file containing textline location+transcription for testing. You can use '
                               'multiple input files quoted and space separated "test_file1.txt test_file2.txt"to '
                               'combine testing sets.')

    # Inference args
    inference_args = parser.add_argument_group('General inference arguments')
    inference_args.add_argument('--do_inference', help='inference',
                                action='store_true')
    inference_args.add_argument('--inference_list', metavar='inference_list', type=str, default=None,
                                help='use this file containing textline location+transcription for inferencing. You can use '
                                'multiple input files quoted and space separated "inference_file1.txt '
                                'inference_file2.txt"to combine inferencing sets.')

    # Model args
    model_args = parser.add_argument_group('Model-specific arguments')
    model_args.add_argument('--model', metavar='model ', type=str, default=None,
                            help='Model to use')
    model_args.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                            help='continue training/validation/testing/inferencing from this model as a starting point.')
    model_args.add_argument('--model_name', metavar='model_name ', type=str, default=None,
                            help='use model_name in the output')
    model_args.add_argument('--use_gru', help='use GRU Gated Recurrent Units instead of LSTM in the recurrent layers',
                            action='store_true')
    model_args.add_argument('--batch_normalization',
                            help='batch_normalization', action='store_true')
    model_args.add_argument('--use_dropout', help='if enabled some dropout will be added to the model if creating a new '
                            'model', action='store_true')
    model_args.add_argument('--use_rnn_dropout', help='if enabled some dropout will be added to rnn layers of the model '
                            'if creating a new model', action='store_true')
    model_args.add_argument('--rnn_layers', metavar='rnn_layers ', type=int, default=5,
                            help='number of rnn layers to use in the recurrent part. default 5')
    model_args.add_argument('--rnn_units', metavar='rnn_units ', type=int, default=256,
                            help='numbers of units in each rnn_layer. default 256')
    model_args.add_argument('--replace_final_layer', action='store_true',
                            help='beta: replace_final_layer. You can do this to extend/decrease the character set when '
                            'using an existing model')
    model_args.add_argument('--replace_recurrent_layer', action='store_true',
                            help='beta: replace_recurrent_layer. Set new recurrent layer using an existing model. '
                            'Additionally replaces final layer as well.')
    model_args.add_argument('--thaw', action='store_true',
                            help='beta: thaw. thaws conv layers, only usable with existing_model')
    model_args.add_argument('--freeze_conv_layers', action='store_true',
                            help='beta: freeze_conv_layers. Freezes conv layers, only usable with existing_model')
    model_args.add_argument('--freeze_recurrent_layers', action='store_true',
                            help='beta: freeze_recurrent_layers. Freezes recurrent layers, only usable with existing_model')
    model_args.add_argument('--freeze_dense_layers', action='store_true',
                            help='beta: freeze_dense_layers. Freezes dense layers, only usable with existing_model')
    model_args.add_argument('--dropout_rnn', type=float, default=0.5,
                            help='beta: dropout_rnn. Default 0.5. Only used when use_dropout_rnn is enabled')
    model_args.add_argument('--dropout_recurrent_dropout', type=float, default=0,
                            help='beta: dropout_recurrent_dropout. Default 0. This is terribly slow on GPU as there is no support in cuDNN RNN ops')
    model_args.add_argument('--reset_dropout', action='store_true',
                            help='beta: reset_dropout')
    model_args.add_argument('--set_dropout', type=float, default=0.5,
                            help='beta: set_dropout')
    model_args.add_argument('--dropout_dense', type=float, default=0.5,
                            help='beta: dropout_dense')
    model_args.add_argument('--dropoutconv', type=float, default=0.0,
                            help='beta: set_dropout')
    model_args.add_argument('--cnn_multiplier', type=int, default=4,
                            help='beta: cnn_multiplier')

    # Data augmentation args
    augmentation_args = parser.add_argument_group('Augmentation arguments')
    augmentation_args.add_argument('--augment', action='store_true',
                                   help='beta: apply data augmentation to training set. In general this is a good idea')

    augmentation_args.add_argument('--elastic_transform', action='store_true',
                                   help='beta: elastic_transform, currently disabled')
    augmentation_args.add_argument('--random_crop', action='store_true',
                                   help='beta: broken. random_crop')
    augmentation_args.add_argument('--random_width', action='store_true',
                                   help='data augmentation option: random_width, stretches the textline horizontally to random width')
    augmentation_args.add_argument('--distort_jpeg', action='store_true',
                                   help='beta: distort_jpeg')
    augmentation_args.add_argument('--do_random_shear', action='store_true',
                                   help='beta: do_random_shear')

    # Word Beam Search arguments
    wbs_args = parser.add_argument_group('Word Beam Search arguments')
    wbs_args.add_argument(
        '--greedy', help='use greedy ctc decoding. beam_width will be ignored', action='store_true')
    wbs_args.add_argument('--beam_width', metavar='beam_width ', type=int, default=10,
                          help='beam_width when validating/inferencing, higher beam_width gets better results, but run '
                          'slower. Default 10')
    wbs_args.add_argument('--num_oov_indices', metavar='num_oov_indices ', type=int, default=0,
                          help='num_oov_indices, default 0, set to 1 if unknown characters are in dataset, but not in '
                          'charlist. Use when you get the error "consider setting `num_oov_indices=1`"')
    wbs_args.add_argument('--corpus_file', metavar='corpus_file ', type=str, default=None,
                          help='beta: corpus_file to use, enables WordBeamSearch')
    wbs_args.add_argument('--wbs_smoothing', metavar='corpus_file ', type=float, default=0.1,
                          help='beta: smoothing to use when using word beam search')

    # Miscellaneous
    misc_parser = parser.add_argument_group('Miscellaneous arguments')

    # parser.add_argument('--num_workers', metavar='num_workers ', type=int, default=20,
    #                     help='num_workers')
    misc_parser.add_argument('--do_binarize_otsu', action='store_true',
                             help='beta: do_binarize_otsu')
    misc_parser.add_argument('--do_binarize_sauvola', action='store_true',
                             help='beta: do_binarize_sauvola')
    misc_parser.add_argument('--ignore_lines_unknown_character', action='store_true',
                             help='beta: ignore_lines_unknown_character. Ignores during training/validation lines that '
                             'contain characters that are not in charlist.')
    misc_parser.add_argument('--check_missing_files', action='store_true',
                             help='beta: check_missing_files')
    misc_parser.add_argument('--normalize_text', action='store_true',
                             help='')
    misc_parser.add_argument('--use_lmdb', action='store_true',
                             help='use lmdb to store images, this might be faster for more epochs')
    misc_parser.add_argument('--reuse_old_lmdb_train', type=str,
                             help='path of the folder of lmdb for training data')
    misc_parser.add_argument('--reuse_old_lmdb_val', type=str,
                             help='path of the folder of lmdb for validation data')
    misc_parser.add_argument('--reuse_old_lmdb_test', type=str,
                             help='path of the folder of lmdb for test data')
    misc_parser.add_argument('--reuse_old_lmdb_inference', type=str,
                             help='path of the folder of lmdb for inference data')
    misc_parser.add_argument('--deterministic', action='store_true',
                             help='beta: deterministic mode (reproducible results')
    misc_parser.add_argument('--no_auto', action='store_true',
                             help='No Auto disabled automatic "fixing" of certain parameters')

    return parser


def fix_args(args):
    if not args.no_auto and args.train_list:
        print('do_train implied by providing a train_list')
        args.__dict__['do_train'] = True
    if not args.no_auto and args.batch_size > 1:
        print('batch_size > 1, setting use_mask=True')
        args.__dict__['use_mask'] = True


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    # TODO: use config
    dictionary = args.__dict__
    fix_args(args)
    print(dictionary)

    return args
