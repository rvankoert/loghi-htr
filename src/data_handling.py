# Imports

# > Standard library
import logging
import os

# > Local dependencies
from data_loader import DataLoader


def initialize_data_loader(args, char_list, model):
    model_height = model.layers[0].input_shape[0][2]
    model_channels = model.layers[0].input_shape[0][3]
    img_size = (model_height, args.width, model_channels)

    return DataLoader(
        batch_size=args.batch_size,
        img_size=img_size,
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
        normalization_file=args.normalization_file,
        use_mask=args.use_mask,
        do_random_shear=args.do_random_shear
    )


def load_initial_charlist(charlist_location, existing_model,
                          output_directory, replace_final_layer):
    # Set the character list location
    if not charlist_location and existing_model:
        charlist_location = existing_model + '/charlist.txt'
    elif not charlist_location:
        charlist_location = output_directory + '/charlist.txt'

    # Load the character list
    char_list = []

    # We don't need to load the charlist if we are replacing the final layer
    if not replace_final_layer:
        if os.path.exists(charlist_location):
            with open(charlist_location) as file:
                char_list = [char for char in file.read()]
            logging.info(f"Using charlist from: {charlist_location}")
        else:
            raise FileNotFoundError(
                f"Charlist not found at: {charlist_location} and "
                "replace_final_layer is False. Exiting...")

        logging.info(f"Using charlist: {char_list}")
        logging.info(f"Charlist length: {len(char_list)}")

    return char_list


def save_charlist(charlist, output, output_charlist_location=None):
    # Save the new charlist
    if not output_charlist_location:
        output_charlist_location = output + '/charlist.txt'
    with open(output_charlist_location, 'w') as chars_file:
        chars_file.write(str().join(charlist))
