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

    # Training args
    training_args = parser.add_argument_group('General training arguments')

    # LR Schedule

    training_args.add_argument('--test_list', metavar='test_list', type=str, default=None,
                               help='use this file containing textline location+transcription for testing. You can use '
                               'multiple input files quoted and space separated "test_file1.txt test_file2.txt"to '
                               'combine testing sets.')

    # Word Beam Search arguments

    # Miscellaneous
    misc_args = parser.add_argument_group('Miscellaneous arguments')

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
    if args.height:
        logger.warning("Argument will lose support in March 2024: --height. "
                       "Height will be inferred from the VGSL spec.")
    if args.channels:
        logger.warning("Argument will lose support in March 2024: --channels. "
                       "Channels will be inferred from the VGSL spec.")


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    arg_future_warning(args)

    # TODO: use config
    fix_args(args)

    return args
