import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#             "recommended":
#                 ("None,64,None,1 Cr3,3,24 Mp2,2,2,2 Bn Cr3,3,48 Bn Cr3,3,96 "
#                  "Mp2,2,2,2 Bn Cr3,3,96 Mp2,2,2,2 Bn Rc Bl512 D50 Bl512 D50 "
#                  "Bl512 D50 Bl512 D50 Bl512 D50 O1s92")
def build_custom_model(img_size, number_of_characters):
    (height, width, channels) = img_size[0], img_size[1], img_size[2]
    padding = "same"
    activation = "relu"
    width = None
    input_img = layers.Input(
        shape=(width, height, channels), name="image"
    )

    use_gru = False
    rnn_layers = 5
    rnn_units = 128
    dropout = False
    use_rnn_dropout = True
    dropoutdense = 0.5
    dropoutconv = 0.0
    dropout_rnn = 0.5

    initializer = tf.keras.initializers.GlorotNormal()
    channel_axis = -1

    x = input_img

    x = layers.Conv2D(
        filters=24,
        kernel_size=[3, 3],
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv1a",
        kernel_initializer=initializer
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    if dropout:
        x = layers.Dropout(dropoutconv)(x)

    x = layers.Conv2D(
        filters=24,
        kernel_size=[3, 1],
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv1b",
        kernel_initializer=initializer
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    if dropout:
        x = layers.Dropout(dropoutconv)(x)

    # Second conv block
    x = layers.Conv2D(
        filters=36,
        kernel_size=[1, 3],
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv2a",
        kernel_initializer=initializer
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Conv2D(
        filters=48,
        kernel_size=[3, 1],
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv2b",
        kernel_initializer=initializer
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    if dropout:
        x = layers.Dropout(dropoutconv)(x)

    x = layers.Conv2D(
        64,
        (1, 3),
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv3a",
        kernel_initializer=initializer
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Conv2D(
        96,
        (3, 1),
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv3b",
        kernel_initializer=initializer
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    if dropout:
        x = layers.Dropout(dropoutconv)(x)

    x = layers.Conv2D(
        96,
        (1, 3),
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv4a",
        kernel_initializer=initializer
    )(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.Conv2D(
        96,
        (3, 1),
        strides=(1, 1),
        activation=activation,
        padding=padding,
        name="Conv4b",
        kernel_initializer=initializer
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
    if batch_normalization:
        x = layers.BatchNormalization(axis=channel_axis)(x)
    if dropout:
        x = layers.Dropout(dropoutconv)(x)

    new_shape = (-1, x.shape[-2] * x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    for i in range(1, rnn_layers + 1):
        if use_gru:
            recurrent = layers.GRU(
                units=rnn_units,
                # activation=activation,
                recurrent_activation="sigmoid",
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                return_sequences=True,
                kernel_initializer=initializer,
                reset_after=True,
                name=f"gru_{i}",
            )
        else:
            recurrent = layers.LSTM(rnn_units,
                                    # activation=activation,
                                    return_sequences=True,
                                    kernel_initializer=initializer,
                                    name=f"lstm_{i}"
                                    )

        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if use_rnn_dropout:
            if i < rnn_layers:
                x = layers.Dropout(rate=dropout_rnn)(x)

    # x = layers.Dense(1024, activation="elu",
    #                  kernel_initializer=initializer)(x)
    if dropout:
        x = layers.Dropout(dropoutdense)(x)

    # Output layer
    x = layers.Dense(number_of_characters + 2, activation="softmax", name="dense3",
                         kernel_initializer=initializer)(x)
    output = layers.Activation('linear', dtype=tf.float32)(x)
    model = keras.models.Model(
        inputs=[input_img], outputs=output, name="model_new13"
    )
    return model