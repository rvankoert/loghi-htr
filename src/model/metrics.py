# Imports

# > Third-party dependencies
import tensorflow as tf
from tensorflow.keras import backend as K


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """

    def __init__(self, name='CER_metric', greedy=True, beam_width=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(
            name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")
        self.greedy = greedy
        self.beam_width = beam_width

    @tf.function
    def update_state(self, y_true, y_pred):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        decode, _ = K.ctc_decode(y_pred,
                                 input_length,
                                 greedy=True)

        decode = K.ctc_label_dense_to_sparse(
            decode[0], K.cast(input_length, 'int32'))

        # Ugly hack to disregard OOV. See CTCLoss for more details
        y_true = tf.where(y_true > 0, y_true - 1, y_true)

        y_true_sparse = K.ctc_label_dense_to_sparse(
            y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, 0))
        y_true_sparse = tf.sparse.retain(
            y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=False)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(
            K.cast(tf.size(y_true_sparse.values), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Error Rate
    """

    def __init__(self, name='WER_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(
            name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, _ = K.ctc_decode(y_pred,
                                 input_length,
                                 greedy=True)

        decode = K.ctc_label_dense_to_sparse(
            decode[0], K.cast(input_length, 'int32'))

        # Ugly hack to disregard OOV. See CTCLoss for more details
        y_true = tf.where(y_true > 0, y_true - 1, y_true)

        y_true_sparse = K.ctc_label_dense_to_sparse(
            y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        y_true_sparse = tf.sparse.retain(
            y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        correct_words_amount = tf.reduce_sum(
            tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(K.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_state(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)
