# Imports

# > Standard Library
import logging
import os

# > Local dependencies

# > Third party libraries
import tensorflow as tf


class LoghiCustomCallback(tf.keras.callbacks.Callback):

    previous_loss = float('inf')

    def __init__(self, save_best=True, save_checkpoint=True, output='output',
                 charlist=None, config=None):
        self.save_best = save_best
        self.save_checkpoint = save_checkpoint
        self.output = output
        self.charlist = charlist
        self.config = config

    def save_model(self, subdir):
        outputdir = os.path.join(self.output, subdir)
        os.makedirs(outputdir, exist_ok=True)
        self.model.save(outputdir + '/model.keras')
        with open(os.path.join(outputdir, 'charlist.txt'), 'w') as chars_file:
            chars_file.write(str().join(self.charlist))
        if self.config is not None:
            self.config.save(outputdir + '/config.json')

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"The average CER for epoch {epoch} is "
                     f"{logs['CER_metric']:7.2f}")

        if logs["val_CER_metric"] is not None:
            current_loss = logs["val_CER_metric"]
        else:
            current_loss = logs["CER_metric"]

        if self.save_best:
            if self.previous_loss is None or self.previous_loss > current_loss:
                logging.info(f"CER has improved from {self.previous_loss:7.2f}"
                             f" to {current_loss:7.2f}")
                self.previous_loss = current_loss
                self.save_model('best_val')
        if self.save_checkpoint:
            if logs["val_CER_metric"]:
                loss_part = f"_val_CER_metric{logs['val_CER_metric']}"
            else:
                loss_part = f"_CER_metric{logs['CER_metric']}"

            logging.info('Saving checkpoint...')
            self.save_model('epoch_' + str(epoch) + loss_part)
            logging.info('Checkpoint saved')
