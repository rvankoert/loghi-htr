# Imports

# > Standard Library
import os
import json

# > Local dependencies

# > Third party libraries
import tensorflow as tf


class LoghiCustomCallback(tf.keras.callbacks.Callback):

    previous_loss = float('inf')

    def __init__(self, save_best=True, save_checkpoint=True, output='output',
                 charlist=None, metadata=None):
        self.save_best = save_best
        self.save_checkpoint = save_checkpoint
        self.output = output
        self.charlist = charlist
        self.metadata = metadata

    def save_model(self, subdir):
        outputdir = os.path.join(self.output, subdir)
        os.makedirs(outputdir, exist_ok=True)
        self.model.save(outputdir + '/model.keras')
        with open(os.path.join(outputdir, 'charlist.txt'), 'w') as chars_file:
            chars_file.write(str().join(self.charlist))
        if self.metadata is not None:
            with open(os.path.join(outputdir, 'config.json'), 'w') as file:
                file.write(json.dumps(self.metadata))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average cer for epoch {} is {:7.2f} "
            .format(
                epoch, logs["CER_metric"]
            )
        )
        if logs["val_CER_metric"] is not None:
            current_loss = logs["val_CER_metric"]
        else:
            current_loss = logs["CER_metric"]

        if self.save_best:
            if self.previous_loss is None or self.previous_loss > current_loss:
                print('cer has improved from {:7.2f} to {:7.2f}'.format(
                    self.previous_loss, current_loss))
                self.previous_loss = current_loss
                self.save_model('best_val')
        if self.save_checkpoint:
            if logs["val_CER_metric"]:
                loss_part = "_val_CER_metric"+str(logs["val_CER_metric"])
            else:
                loss_part = "_CER_metric" + str(logs["CER_metric"])
            print('saving checkpoint')
            self.save_model('epoch_' + str(epoch) + loss_part)
