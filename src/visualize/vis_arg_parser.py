# Imports

# > Standard Library
import argparse

# > Local dependencies

# > Third party libraries

def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Loghi-htr visualization '
                    'for visualizing model weights/layers')

    # General args (visualize_filters_activations.py)
    parser.add_argument('--seed', metavar='seed', type=int, default=42,
                        help='random seed to be used')
    parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                        help='gpu to be used')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1')
    parser.add_argument('--height', metavar='height', type=int, default=64,
                        help='height to be used')
    parser.add_argument('--num_filters_per_row', metavar='num_filters_per_row', type=int, default=8,
                        help='amount of filter plots to make per row (visualize_filters_activations.py specific)')
    parser.add_argument('--channels', metavar='channels', type=int, default=4,
                        help='channels to be used')
    parser.add_argument('--width', metavar='width', type=int, default=751,
                        help='width to be used')
    parser.add_argument('--output', metavar='output', type=str, default='visualize_plots',
                        help='base output to be used')
    parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                        help='percent_validation to be used')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.00001,
                        help='learning_rate to be used')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                        help='epochs to be used')
    parser.add_argument('--dataloader_img_shape', metavar='dataloader_img_shape', type=tuple, default=(128,128,4),
                        help='height, width and channels to be used for dataloader')
    parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                        help='dataset. ecodices or iisg')
    parser.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                        help='validation_list')
    parser.add_argument('--sample_image', metavar='sample_image', type=str, default="",
                        help='single png to for saliency plots')
    parser.add_argument('--light_mode', action='store_true', default=False,
                        help='for setting the output image background + font color')
    parser.add_argument('--do_binarize_otsu', action='store_true', default = False,
                        help='prefix to use for testing')
    parser.add_argument('--do_binarize_sauvola', action='store_true',
                        help='do_binarize_sauvola')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                        help='existing_model')
    parser.add_argument('--do_detailed', action='store_true', default=False, help="param for making more "
                                                                                               "detailed "
                                                                                               "visualizations (at "
                                                                                               "the cost of performace")
    return parser

def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    dictionary = args.__dict__
    print(dictionary)

    return args