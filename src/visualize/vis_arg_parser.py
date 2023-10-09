# Imports

# > Standard Library
import argparse

# > Local dependencies

# > Third party libraries

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Loghi-htr visualization '
                    'for visualizing model weights/layers')

    # General args (visualize_network.py)
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                        help='gpu to be used')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1')
    parser.add_argument('--height', metavar='height', type=int, default=64,
                        help='height to be used')
    parser.add_argument('--channels', metavar='channels', type=int, default=3,
                        help='channels to be used')
    parser.add_argument('--width', metavar='width', type=int, default=751,
                        help='width to be used')
    parser.add_argument('--output', metavar='output', type=str, default='output',
                        help='base output to be used')
    parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                        help='percent_validation to be used')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.00001,
                        help='learning_rate to be used')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                        help='epochs to be used')
    # parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
    #                     help='spec')
    # parser.add_argument('--trainset', metavar='trainset', type=str, default='/data/cvl-database-1-1/train.txt',
    #                     help='trainset to be used')
    # parser.add_argument('--testset', metavar='testset', type=str, default='/data/cvl-database-1-1/test.txt',
    #                     help='testset to be used')
    parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                        help='dataset. ecodices or iisg')
    parser.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                        help='validation_list')
    parser.add_argument('--do_binarize_otsu', action='store_true',
                        help='prefix to use for testing')
    parser.add_argument('--do_binarize_sauvola', action='store_true',
                        help='do_binarize_sauvola')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                        help='existing_model')
    return parser

def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    dictionary = args.__dict__
    print(dictionary)

    return args
