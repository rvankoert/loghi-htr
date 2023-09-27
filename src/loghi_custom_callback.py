import os
from tensorflow import keras
import json

class LoghiCustomCallback(keras.callbacks.Callback):

    previous_loss = float('inf')
    def __init__(self, save_best=True, save_checkpoint=True, output='output', charlist=None, metadata=None):
        self.save_best = save_best
        self.save_checkpoint = save_checkpoint
        self.output = output
        self.charlist = charlist
        self.metadata = metadata

    def save_model(self, subdir):
        outputdir=os.path.join(self.output, subdir)
        self.model.save(outputdir)
        with open(os.path.join(outputdir, 'charlist.txt'), 'w') as chars_file:
            chars_file.write(str().join(self.charlist))
        if self.metadata is not None:
            with open(os.path.join(outputdir, 'config.json'), 'w') as file:
                file.write(json.dumps(self.metadata))

    # def on_train_batch_end(self, batch, logs=None):
    #     print(
    #         "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
    #     )

    # def on_test_batch_end(self, batch, logs=None):
    #     print(
    #         "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
    #     )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
                .format(
                epoch, logs["loss"]
            )
        )
        if logs["val_loss"] is not None:
            current_loss = logs["val_loss"]
        else:
            current_loss = logs["loss"]

        if self.save_best:
            if self.previous_loss is None or self.previous_loss > current_loss:
                print('loss has improved from {:7.2f} to {:7.2f}'.format(self.previous_loss, current_loss))
                self.previous_loss = current_loss
                self.save_model('best_val')
        if self.save_checkpoint:
            if logs["val_loss"]:
                loss_part = "_val_loss"+str(logs["val_loss"])
            else:
                loss_part = "_loss" + str(logs["loss"])
            print('saving checkpoint')
            self.save_model('epoch_' + str(epoch) + loss_part)
