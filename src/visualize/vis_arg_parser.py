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
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                        help='batch_size to be used, when using variable sized input this must be 1')
    parser.add_argument('--num_filters_per_row', metavar='num_filters_per_row', type=int, default=5,
                        help='amount of filter plots to make per row (visualize_filters_activations.py specific)')
    parser.add_argument('--sample_image_path', metavar='sample_image_path', type=str, default="",
                        help='single png to for saliency plots')
    parser.add_argument('--light_mode', action='store_true', default=False,
                        help='for setting the output image background + font color')
    parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                        help='existing_model')
    parser.add_argument('--replace_header', metavar='replace_header ', type=str, default='',
                        help='replace_header')
    parser.add_argument("--do_detailed", action='store_true', default=False, help="param for making more detailed "
                                                                                  "visualizations (at the cost of "
                                                                                  "performace")
    return parser


def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()
    return args
