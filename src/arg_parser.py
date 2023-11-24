# Imports

# > Standard Library
import argparse
import logging

# > Local dependencies

# > Third party libraries


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
    general_args.add_argument('--max_queue_size', metavar='max_queue_size ', type=int, default=256,
                              help='max_queue_size')
    general_args.add_argument('--charlist', metavar='charlist ', type=str, default=None,
                              help='Charlist to use')
    general_args.add_argument('--output_charlist', metavar='output_charlist', type=str, default=None,
                              help='output_charlist to use')

    # Training args
    training_args = parser.add_argument_group('General training arguments')
    training_args.add_argument('--learning_rate', metavar='learning_rate',
                               type=float, default=0.0003,
                               help='learning_rate to be used, default 0.0003')
    training_args.add_argument('--epochs', metavar='epochs', type=int,
                               default=40, help='epochs to be used, default 40')
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
    training_args.add_argument('--training_verbosity_mode', choices=['auto', '0', '1', '2'], default='auto',
                               help="0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' becomes 1 for most cases. "
                                    "default value is 'auto'")

    # Inference args
    inference_args = parser.add_argument_group('General inference arguments')
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
    model_args.add_argument('--replace_final_layer', action='store_true',
                            help='beta: replace_final_layer. You can do this to extend/decrease the character set when '
                            'using an existing model')
    model_args.add_argument('--replace_recurrent_layer', action='store',
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
    misc_args = parser.add_argument_group('Miscellaneous arguments')

    misc_args.add_argument('--do_binarize_otsu', action='store_true',
                           help='beta: do_binarize_otsu')
    misc_args.add_argument('--do_binarize_sauvola', action='store_true',
                           help='beta: do_binarize_sauvola')
    misc_args.add_argument('--ignore_lines_unknown_character', action='store_true',
                           help='beta: ignore_lines_unknown_character. Ignores during training/validation lines that '
                           'contain characters that are not in charlist.')
    misc_args.add_argument('--check_missing_files', action='store_true',
                           help='beta: check_missing_files')
    misc_args.add_argument('--normalization_file', default=None, type=str,
                           help='The location of a json file that contains the characters to be normalized. The keys '
                           'are the characters to be replaced, the values are the characters to replace with.')
    misc_args.add_argument('--deterministic', action='store_true',
                           help='beta: deterministic mode (reproducible results')
    misc_args.add_argument('--do_blur', action='store_true',
                           help='blur the images for training purposes')
    misc_args.add_argument('--do_invert', action='store_true',
                           help='use with images that have light ink and dark background')

    # Deprecation zone
    depr_args = parser.add_argument_group(
        'Deprecation zone', 'These arguments will be removed in the future')
    depr_args.add_argument('--do_train', help='enable the training. '
                                              'Use this flag if you want to train.',
                           action='store_true')
    depr_args.add_argument('--do_inference', help='inference',
                           action='store_true')
    depr_args.add_argument(
        '--use_mask', help='whether or not to mask certain parts of the data. Defaults to true when batch_size > 1', action='store_true')
    depr_args.add_argument('--no_auto', action='store_true',
                           help='No Auto disabled automatic "fixing" of certain parameters')

    return parser


def fix_args(args):
    if not args.no_auto and args.train_list:
        print('do_train implied by providing a train_list')
        args.__dict__['do_train'] = True
    if not args.no_auto and args.batch_size > 1:
        print('batch_size > 1, setting use_mask=True')
        args.__dict__['use_mask'] = True


def arg_future_warning(args):
    logger = logging.getLogger(__name__)

    # March 2024
    if args.do_train:
        logger.warning("Argument will lose support in March 2024: --do_train. "
                       "Training will be enabled by providing a train_list. ")
    if args.do_inference:
        logger.warning("Argument will lose support in March 2024: "
                       "--do_inference. Inference will be enabled by "
                       "providing an inference_list. ")
    if args.use_mask:
        logger.warning("Argument will lose support in March 2024: --use_mask. "
                       "Masking will be enabled by default.")
    if args.no_auto:
        logger.warning("Argument will lose support in March 2024: --no_auto.")


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    arg_future_warning(args)

    # TODO: use config
    fix_args(args)

    return args
