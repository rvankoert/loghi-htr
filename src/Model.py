import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # print("CTC lambda inputs / shape")
        # print("y_pred:", y_pred.shape)  # (?, 778, 30)
        # print("labels:", y_true.shape)  # (?, 80)
        # print("input_length:", input_length.shape)  # (?, 1)
        # print("label_length:", label_length.shape)  # (?, 1)
        # print("loss:", loss)  # (?, 1)

        # At test time, just return the computed predictions
        return y_pred


# class CERMetric(tf.keras.metrics.Metric):
#     """
#     A custom Keras metric to compute the Character Error Rate
#     """
#
#     def __init__(self, name='CER_metric', **kwargs):
#         super(CERMetric, self).__init__(name=name, **kwargs)
#         self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
#         self.counter = self.add_weight(name="cer_count", initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         input_shape = K.shape(y_pred)
#         input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
#
#         decode, log = K.ctc_decode(y_pred,
#                                    input_length,
#                                    greedy=True)
#
#         decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
#         y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
#
#         decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
#         distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
#
#         self.cer_accumulator.assign_add(tf.reduce_sum(distance))
#         self.counter.assign_add(len(y_true))
#
#     def result(self):
#         return tf.math.divide_no_nan(self.cer_accumulator, self.counter)
#
#     def reset_states(self):
#         self.cer_accumulator.assign(0.0)
#         self.counter.assign(0.0)
#
#
# class WERMetric(tf.keras.metrics.Metric):
#     """
#     A custom Keras metric to compute the Word Error Rate
#     """
#
#     def __init__(self, name='WER_metric', **kwargs):
#         super(WERMetric, self).__init__(name=name, **kwargs)
#         self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
#         self.counter = self.add_weight(name="wer_count", initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         input_shape = K.shape(y_pred)
#         input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
#
#         decode, log = K.ctc_decode(y_pred,
#                                    input_length,
#                                    greedy=True)
#
#         decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
#         y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
#
#         decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
#         distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
#
#         correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))
#
#         self.wer_accumulator.assign_add(correct_words_amount)
#         self.counter.assign_add(len(y_true))
#
#     def result(self):
#         return tf.math.divide_no_nan(self.wer_accumulator, self.counter)
#
#     def reset_states(self):
#         self.wer_accumulator.assign(0.0)
#         self.counter.assign(0.0)


class Model():

    def build_model(self, imgSize, number_characters, learning_rate):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutdense = 0.5
        dropoutconv = 0.1
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        labels = layers.Input(name="label", shape=(None,))

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1,1),
            activation='elu',
            padding=padding,
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1,1),
            activation='elu',
            padding=padding,
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1,1),
            activation='elu',
            padding=padding,
            name="Conv3",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1,1),
            activation='elu',
            padding=padding,
            name="Conv4",
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            80,
            (3, 3),
            strides=(1,1),
            activation='elu',
            padding=padding,
            name="Conv5",
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 80)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(1024, activation="elu", name="dense1")(x)
        x = layers.Dropout(dropoutdense)(x)
        x = layers.Dense(1024, activation="elu", name="dense2")(x)
        x = layers.Dropout(dropoutdense)(x)

        # x = tf.keras.layers.Masking(mask_value=0)(x)
        # x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        # x = tf.keras.layers.Masking(mask_value=0)(x)
        # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

        # Output layer
        x = layers.Dense(number_characters + 1, activation="softmax", name="dense3")(x)

        x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    #
    # # Train the model
    def train_batch(self, model, train_dataset, validation_dataset, epochs, filepath):
        early_stopping_patience = 50
        # # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        )
        from keras.callbacks import History
        from keras.callbacks import ModelCheckpoint
        history = History()
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            # batch_size=1,
            callbacks=[early_stopping, history, checkpoint],
            shuffle=True,
            workers=16,
            max_queue_size=256
        )
        return history
#
# # Get the prediction model by extracting layers till the output layer
# prediction_model = keras.models.Model(
#     model.get_layer(name="image").input, model.get_layer(name="dense2").output
# )
# prediction_model.summary()
#
#
# # A utility function to decode the output of the network
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
#               :, :max_length
#               ]
#     # Iterate over the results and get back the text
#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text
#
