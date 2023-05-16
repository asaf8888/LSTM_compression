import re
import numpy
import os
from prediction_model.my_model import MyModel
from prediction_model.quantizable_model import get_quantizable_model, convert_to_tflite
from prediction_model.model_constants import *
from compression.huffman import extract_all_token_amount
from compression.compression_constants import unknown_character_token
import pathlib


def prepare_training_dataset(text, vocab, seq_lentgh=100, batch_size=64):
    text = [list(x) for x in text]
    text = list(numpy.concatenate(text).flat)
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)

    ids_dataset = tf.data.Dataset.from_tensor_slices(ids_from_chars(text))
    sequences = ids_dataset.batch(seq_lentgh + 1, drop_remainder=True)
    dataset = sequences.map(lambda x: (x[:-1], x[1:]))

    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset


def get_trained_model(filepath, unknown_token_cutoff=0):
    raw_file = open(filepath, 'r')
    text = raw_file.readlines()
    raw_file.close()

    vocab, unknown = extract_all_token_amount(text, unknown_token_cutoff)
    vocab[unknown_character_token] = unknown[1]

    dataset = prepare_training_dataset(text, vocab)



vocab_size = len(vocab)
print(f"vocab size: {vocab_size}")


model = MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics='accuracy')
history = model.fit(dataset, epochs=EPOCHS)
quant_model = get_quantizable_model(model)

with open("../quantized_bible_model/model_quant.tflite", 'wb') as f:
    f.write(convert_to_tflite(quant_model))


