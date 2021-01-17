from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import cv2
import editdistance
from Model import Model
from DataLoader import DataLoader
from SamplePreprocessor import preprocess
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    "main function"

    batchSize = 32
    imgSize = (1024, 48, 4)
    maxTextLen = 128
    epochs = 1000
    learning_rate = 0.001
    # load training data, create TF model
    loader = DataLoader(FilePaths.fnTrain, batchSize, imgSize, maxTextLen)

    # save characters of model for inference mode
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
    # print(loader.charList)

    modelClass = Model()
    print(len(loader.charList))
#    model = keras.models.load_model('../models/model-val-best-52.96145')
    model = modelClass.build_model(imgSize, len(loader.charList), learning_rate)  # (loader.charList, keep_prob=0.8)
#    model.compile(keras.optimizers.Adam(learning_rate=learning_rate))
    model.summary()

    batch = loader.getTrainDataSet()
    validation_dataset = loader.getValidationDataSet()
    print("batch")
    print(batch)
    print("batch")
    print(validation_dataset)

    best_loss = 999999
    best_val_loss = 999999

    # for epoch in range(0, epochs):
    #     # history = Model().train_batch(model, batch, validation_dataset, epochs=epochs)
    #     print("starting epoch "+str(epoch)+"/"+str(epochs))
    history = Model().train_batch(model, batch, validation_dataset, epochs=epochs, filepath='../models/model-val-best')
    #     model.save('../models/model-full-epoch' + str(epoch))
    #     val_loss = history.history.get('val_loss')[0]
    #     loss = history.history.get('loss')[0]
    #     print('Loss:', loss)
    #     print('val_loss:', val_loss)
    #     if loss < best_loss:
    #         best_loss = loss
    #         model.save('../models/best-train-model-' + str(loss))
    #         print('new best trainModel:', loss)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         model.save('../models/best-val-model-' + str(val_loss))
    #         print('new best valModel:', val_loss)

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense3").output
    )
    prediction_model.summary()

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :maxTextLen
                  ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(loader.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    #  Let's check results on some validation samples
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(loader.num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label.strip())

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            # for i in range(16):
            print(orig_texts[i].strip())
            print(pred_texts[i].strip())


# 		img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
# 		img = img.T
# 		title = f"Prediction: {pred_texts[i].strip()}"
# 		ax[i // 4, i % 4].imshow(img, cmap="gray")
# 		ax[i // 4, i % 4].set_title(title)
# 		ax[i // 4, i % 4].axis("off")
# plt.show()

if __name__ == '__main__':
    main()
