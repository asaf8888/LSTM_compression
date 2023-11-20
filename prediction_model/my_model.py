import keras
import tensorflow as tf
from enum import Enum
class SingleUseModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states, carry = self.rnn(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, (states, carry)
        else:
            return x


class SplitReadyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, name="lstm")
        self.dense1 = tf.keras.layers.Dense(rnn_units, activation="relu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states, carry = self.rnn(x, initial_state=states, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        if return_state:
            return x, (states, carry)
        else:
            return x

    def split(self):
        model_a = SplitFirstHalf(self.embedding, self.rnn, self.dense1)
        model_b = SplitSecondHalf(self.dense2)
        return model_a, model_b

class SplitFirstHalf(keras.Model):
    def __init__(self, embedding, rnn, dense1):
        super().__init__(self)
        self.embedding = embedding
        self.rnn = rnn
        self.dense1 = dense1


    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.rnn.get_initial_state(x)
        x, states, carry = self.rnn(x, initial_state=states, training=training)
        x = self.dense1(x, training=training)

        if return_state:
            return x, (states, carry)
        else:
            return x
class SplitSecondHalf(keras.Model):
    def __init__(self, dense2):
        super().__init__(self)
        self.dense2 = dense2

    def call(self, x, training=False):

        x = self.dense2(x, training=training)
        return x

class ModelFactory:
    class ModelType(Enum):
        SINGLE_USE_MODEL = 1
        SPLIT_READY_MODEL = 2

    def __init__(self, model_parameters, model_type):
        self.model_parameters = model_parameters
        self.model_type = model_type

    def get_single_use_model(self, vocab_size):
        model_parameters = self.model_parameters
        return SingleUseModel(vocab_size=vocab_size, embedding_dim=model_parameters.embedding_dim,
                               rnn_units=model_parameters.rnn_units)

    def get_split_ready_model(self, vocab_size):
        model_parameters = self.model_parameters
        return SplitReadyModel(vocab_size=vocab_size, embedding_dim=model_parameters.embedding_dim,
                              rnn_units=model_parameters.rnn_units)

    def get_model(self, vocab_size):
        if self.model_type == self.ModelType.SINGLE_USE_MODEL:
            return self.get_single_use_model(vocab_size)
        elif self.model_type == self.ModelType.SPLIT_READY_MODEL:
            return self.get_split_ready_model(vocab_size)
        else:
            raise Exception("missing or unknown model type configuration")

