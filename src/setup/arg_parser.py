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

    # General configuration
    general_args = parser.add_argument_group('General arguments')
    general_args.add_argument('--gpu', metavar='gpu', type=str, default="0",
                              help="Specify the GPU ID for training/inference."
                              " Use -1 to run on CPU. Default: '0'.")
    general_args.add_argument('--output', metavar='output', type=str,
                              default='output', help="Base directory for "
                              "saving output files such as models and logs. "
                              "Default: 'output'.")
    general_args.add_argument('--config_file', metavar='config_file', type=str,
                              help="Path to a JSON file containing arguments "
                              "for the model.")
    general_args.add_argument('--batch_size', metavar='batch_size', type=int,
                              default=4, help="Number of samples processed in "
                              "one iteration. Affects memory usage and "
                              "training speed. Default: 4.")
    general_args.add_argument('--seed', metavar='seed', type=int, default=42,
                              help="Seed for random number generators to "
                              "ensure reproducibility. Default: 42.")
    general_args.add_argument('--charlist', metavar='charlist ', type=str,
                              default=None, help="Path to a file containing "
                              "the list of characters to be recognized. "
                              "Required for inference and validation.")
    general_args.add_argument('--test_list', metavar='test_list',
                              type=str, default=None, help="File(s) with "
                              "textline locations and transcriptions for "
                              "testing. For multiple files, separate with "
                              "spaces and use quotes: 'test_file1.txt "
                              "test_file2.txt'.")

    # Training configuration
    training_args = parser.add_argument_group('Training arguments')
    training_args.add_argument('--epochs', metavar='epochs', type=int,
                               default=40, help="Number of training epochs. "
                               "Default: 40.")
    training_args.add_argument('--width', metavar='width', type=int,
                               default=65536, help="Maximum image width for "
                               "training. Generally, does not need to be "
                               "changed. Default: 65536.")
    training_args.add_argument('--train_list', metavar='train_list', type=str,
                               default=None, help="Path to a file (or "
                               "multiple files) with textline locations and "
                               "transcriptions for training. Use "
                               "space-separated quotes for multiple files: "
                               "'training_file1.txt training_file2.txt'.")
    training_args.add_argument('--steps_per_epoch', metavar='steps_per_epoch',
                               type=int, default=None, help="Number of steps "
                               "per training epoch. Default: None (calculated "
                               "based on dataset size).")
    training_args.add_argument('--output_checkpoints', action='store_true',
                               help="Enable continuous output of checkpoints "
                               "after each epoch. By default, only the best "
                               "validation result is saved.")
    training_args.add_argument('--early_stopping_patience', type=int,
                               default=20, help="Number of epochs with no "
                               "improvement after which training will be "
                               "stopped. Default: 20.")
    training_args.add_argument('--do_validate', action='store_true',
                               help="Enable a separate validation run during "
                               "training.")
    training_args.add_argument('--validation_list', metavar='validation_list',
                               type=str, default=None, help="File(s) "
                               "containing textline locations and "
                               "transcriptions for validation. Format: "
                               "'validation_file1.txt validation_file2.txt' "
                               "for multiple files.")
    training_args.add_argument('--training_verbosity_mode',
                               choices=['auto', '0', '1', '2'], default='auto',
                               help="Set verbosity mode for training output. "
                               "0: Silent, 1: Progress Bar, 2: One Line Per "
                               "Epoch. 'auto' defaults to 1 for most cases. "
                               "Default: 'auto'.")
    training_args.add_argument('--max_queue_size', metavar='max_queue_size',
                               type=int, default=256, help="Maximum size for "
                               "the generator queue. Default: 256.")

    # Inference configuration
    inference_args = parser.add_argument_group('Inference arguments')
    inference_args.add_argument('--inference_list', metavar='inference_list',
                                type=str, default=None, help="File(s) with "
                                "textline locations and transcriptions for "
                                "inference. For multiple files, separate with "
                                "spaces and use quotes: 'inference_file1.txt "
                                "inference_file2.txt'.")
    inference_args.add_argument('--results_file', metavar='results_file',
                                type=str, default='output/results.txt',
                                help="File path to store the results of "
                                "inference. Default: 'output/results.txt'.")

    # Learning rate and optimizer configuration
    lr_args = parser.add_argument_group('Learning rate arguments')
    lr_args.add_argument('--optimizer', metavar='optimizer', type=str,
                         default='adamw', help="Choice of optimizer for "
                         "training. Default: 'adamw'.")
    lr_args.add_argument('--learning_rate', metavar='learning_rate',
                         type=float, default=0.0003,
                         help="Initial learning rate for training. Default: "
                         "0.0003.")
    lr_args.add_argument('--decay_rate', type=float, default=0.99,
                         help="Decay rate for the learning rate. Set to 1 for "
                         "no decay. Default: 0.99.")
    lr_args.add_argument('--decay_steps', type=int, default=-1,
                         help="Number of steps after which the learning rate "
                         "decays. Set to -1 for decay each epoch. Default: -1."
                         )
    lr_args.add_argument('--warmup_ratio', type=float, default=0.0,
                         help="Proportion of total training steps used for "
                         "the warmup phase. Default: 0.0.")
    lr_args.add_argument('--decay_per_epoch', action='store_true',
                         help="Apply learning rate decay per epoch. By "
                         "default, decay is applied per step.")
    lr_args.add_argument('--linear_decay', action='store_true',
                         help="Enable linear decay of the learning rate. If "
                         "not set, exponential decay is used by default.")

    # Model configuration
    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('--model', metavar='model', type=str, default=None,
                            help="Specify the model architecture to be used. "
                            "Default: None (requires specification).")
    model_args.add_argument('--use_float32', action='store_true',
                            help="Use 32-bit float precision in the model. "
                            "Can improve performance at the cost of memory.")
    model_args.add_argument('--existing_model', metavar='existing_model',
                            type=str, default=None, help="Path to an existing "
                            "model to continue training, validation, testing, "
                            "or inferencing. Used as a starting point.")
    model_args.add_argument('--model_name', metavar='model_name', type=str,
                            default=None, help="Custom name for the model, "
                            "used in outputs. Default: None (uses the model "
                            "architecture name).")
    model_args.add_argument('--replace_final_layer', action='store_true',
                            help="Replace the final layer of an existing "
                            "model, useful for adjusting the character set "
                            "size.")
    model_args.add_argument('--replace_recurrent_layer', action='store',
                            help="Replace the recurrent layer in an existing "
                            "model, affecting the final layer as well.")
    model_args.add_argument('--thaw', action='store_true',
                            help="Unfreeze convolutional layers in an "
                            "existing model for further training.")
    model_args.add_argument('--freeze_conv_layers', action='store_true',
                            help="Freeze convolutional layers in an existing "
                            "model, preventing them from updating during "
                            "training.")
    model_args.add_argument('--freeze_recurrent_layers', action='store_true',
                            help="Freeze recurrent layers in an existing "
                            "model to keep them static during training.")
    model_args.add_argument('--freeze_dense_layers', action='store_true',
                            help="Freeze dense layers in an existing model, "
                            "useful for training only certain parts of the "
                            "model.")

    # Data augmentation configuration
    augmentation_args = parser.add_argument_group('Augmentation arguments')
    augmentation_args.add_argument('--aug_multiply', '--multiply',
                                   type=int, default=1,
                                   help="Factor to increase the "
                                   "size of the training dataset by "
                                   "duplicating images. Default: 1 (no "
                                   "duplication).")
    augmentation_args.add_argument('--aug_elastic_transform',
                                   '--elastic_transform', action='store_true',
                                   help="Apply elastic transformations to "
                                        "images.")
    augmentation_args.add_argument('--aug_random_crop', '--random_crop',
                                   action='store_true',
                                   help="Enable random cropping of images.")
    augmentation_args.add_argument('--aug_random_width', '--random_width',
                                   action='store_true',
                                   help="Randomly stretch images horizontally "
                                        "during augmentation.")
    augmentation_args.add_argument('--aug_distort_jpeg', '--distort_jpeg',
                                   action='store_true',
                                   help="Apply JPEG distortion to images for "
                                        "augmentation.")
    augmentation_args.add_argument('--aug_random_shear', '--do_random_shear',
                                   action='store_true',
                                   help="Apply random shearing "
                                        "transformations to images.")
    augmentation_args.add_argument('--aug_blur', '--do_blur',
                                   action='store_true',
                                   help="Apply blurring to images during "
                                        "training for augmentation.")
    augmentation_args.add_argument('--aug_invert', '--do_invert',
                                   action='store_true',
                                   help="Invert images with light ink on dark "
                                        "backgrounds for augmentation.")
    augmentation_args.add_argument('--aug_binarize_otsu',
                                   '--do_binarize_otsu', action='store_true',
                                   help="Apply Otsu's binarization method to "
                                        "images for augmentation.")
    augmentation_args.add_argument('--aug_binarize_sauvola',
                                   '--do_binarize_sauvola',
                                   action='store_true',
                                   help="Use Sauvola's method for image "
                                   "binarization during augmentation.")
    augmentation_args.add_argument('--visualize_augments',
                                   action='store_true',
                                   help='Prompt to create visualization of '
                                        'selected augments')

    # WBS and decoding configuration
    decoding_args = parser.add_argument_group('Decoding arguments')
    decoding_args.add_argument('--greedy', help='use greedy ctc decoding. '
                               'beam_width will be ignored',
                               action='store_true')
    decoding_args.add_argument('--beam_width', metavar='beam_width ', type=int,
                               default=10, help='beam_width when validating/'
                               'inferencing, higher beam_width gets better '
                               'results, but run slower. Default 10')
    decoding_args.add_argument('--corpus_file', metavar='corpus_file',
                               type=str, default=None, help='beta: '
                               'corpus_file to use, enables WordBeamSearch')
    decoding_args.add_argument('--wbs_smoothing', metavar='corpus_file ',
                               type=float, default=0.1, help='beta: smoothing '
                               'to use when using word beam search')

    # Miscellaneous configuration
    misc_args = parser.add_argument_group('Miscellaneous arguments')
    misc_args.add_argument('--normalization_file', default=None, type=str,
                           help="Path to a JSON file specifying character "
                           "normalizations. Format: {'original': "
                           "'replacement'}.")
    misc_args.add_argument('--deterministic', action='store_true',
                           help="Enable deterministic mode for reproducible "
                           "results, at the cost of performance.")

    # Deprecation zone
    depr_args = parser.add_argument_group(
        'Deprecation zone', 'These arguments will be removed in the future')
    depr_args.add_argument('--do_train', help='enable the training. '
                           'Use this flag if you want to train.',
                           action='store_true')
    depr_args.add_argument('--do_inference', help='inference',
                           action='store_true')
    depr_args.add_argument('--use_mask', help='whether or not to mask certain '
                           'parts of the data. Defaults to true when '
                           'batch_size > 1', action='store_true')
    depr_args.add_argument('--no_auto', action='store_true',
                           help='No Auto disabled automatic "fixing" of '
                           'certain parameters')
    depr_args.add_argument('--height', metavar='height', type=int, default=64,
                           help='rescale everything to this height before '
                           'training, default 64')
    depr_args.add_argument('--channels', metavar='channels', type=int,
                           default=3, help='number of channels to use. 1 for '
                           'grey-scale/binary images, three for color images, '
                           '4 for png\'s with transparency')
    depr_args.add_argument('--output_charlist', metavar='output_charlist',
                           type=str, default=None, help="Path to save the "
                           "character list used during training/inference. "
                           "If not specified, the charlist is saved to"
                           "'output/charlist.txt'.")
    depr_args.add_argument('--config_file_output',
                           metavar='config_file_output', type=str,
                           default=None, help="Path to save the "
                           "configuration file. If not specified, the "
                           "configuration is set to 'output/config.json'.")

    return parser


def fix_args(args):
    if not args.no_auto and args.train_list:
        logging.warning('do_train implied by providing a train_list')
        args.__dict__['do_train'] = True
    if not args.no_auto and args.batch_size > 1:
        logging.warning('batch_size > 1, setting use_mask=True')
        args.__dict__['use_mask'] = True
    if not args.no_auto and args.inference_list:
        logging.warning('do_inference implied by providing a inference_list')
        args.__dict__['do_inference'] = True


def arg_future_warning(args):
    logger = logging.getLogger(__name__)

    # May 2024
    if args.do_train:
        logger.warning("Argument will lose support in May 2024: --do_train. "
                       "Training will be enabled by providing a train_list. ")
    if args.do_inference:
        logger.warning("Argument will lose support in May 2024: "
                       "--do_inference. Inference will be enabled by "
                       "providing an inference_list. ")
    if args.use_mask:
        logger.warning("Argument will lose support in May 2024: --use_mask. "
                       "Masking will be enabled by default.")
    if args.no_auto:
        logger.warning("Argument will lose support in May 2024: --no_auto.")
    if args.height:
        logger.warning("Argument will lose support in May 2024: --height. "
                       "Height will be inferred from the VGSL spec.")
    if args.channels:
        logger.warning("Argument will lose support in May 2024: --channels. "
                       "Channels will be inferred from the VGSL spec.")
    if args.output_charlist:
        logger.warning("Argument will lose support in May 2024: "
                       "--output_charlist. The charlist will be saved to "
                       "output/charlist.txt by default.")
    if args.config_file_output:
        logger.warning("Argument will lose support in May 2024: "
                       "--config_file_output. The configuration will be saved "
                       "to output/config.json by default.")


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Determine which arguments were explicitly passed.
    # https://stackoverflow.com/questions/58594956/find-out-which-arguments-were-passed-explicitly-in-argparse
    sentinel = object()

    # Make a copy of args where everything is the sentinel.
    sentinel_ns = argparse.Namespace(**{key: sentinel for key in vars(args)})
    parser.parse_args(namespace=sentinel_ns)

    # Now everything in sentinel_ns that is still the sentinel was not
    # explicitly passed.
    explicit = argparse.Namespace(**{key: (value is not sentinel)
                                     for key, value in
                                     vars(sentinel_ns).items()})

    # TODO: remove after deprecation period
    arg_future_warning(args)
    fix_args(args)

    return args, explicit
