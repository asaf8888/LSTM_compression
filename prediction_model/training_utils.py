import numpy
from prediction_model.my_model import MyModel
from prediction_model.model_constants import *
from compression.compression_constants import unknown_character_token
from collections import Counter
from prediction_model.quantizable_model import get_quantizable_model, convert_to_tflite
import tensorflow as tf


def get_vocabulary_and_mask(text, cutoff=0):
    vocab = dict(Counter(text))
    total = sum(vocab.values())
    cut_token_vocab = {key: value for key, value in vocab.items() if value / total < cutoff}
    unknown = list(cut_token_vocab.keys())
    for key in unknown:
        vocab.pop(key)
    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    keys = list(vocab.keys())
    id_vocab = {keys[i]: i for i in range(len(keys))}
    vocab[unknown_character_token] = len(vocab)
    masked_text = [token if token in vocab.keys() else unknown_character_token for token in text]
    return masked_text, id_vocab, unknown


def prepare_training_dataset(text, vocab, seq_lentgh=100, batch_size=64):
    ids = [vocab[token] for token in text]

    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
    sequences = ids_dataset.batch(seq_lentgh + 1, drop_remainder=True)
    dataset = sequences.map(lambda x: (x[:-1], x[1:]))

    dataset = (
        dataset
        .shuffle(len(dataset))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset


def get_trained_model(text, model_parameters, unknown_token_cutoff=0):
    masked_text, vocab, unknown = get_vocabulary_and_mask(text, unknown_token_cutoff)
    dataset = prepare_training_dataset(masked_text, vocab)
    vocab_size = len(vocab)
    model = MyModel(vocab_size=vocab_size, embedding_dim=model_parameters.embedding_dim, rnn_units=model_parameters.rnn_units)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics='accuracy')
    model.fit(dataset, epochs=EPOCHS)
    return model, (vocab, unknown)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    file = open("../test data/bible.txt", 'r')
    text = file.read()
    file.close()
    model, (vocab, unknown) = get_trained_model(text)
    quant_model = get_quantizable_model(model)

    with open("../quantized_bible_model/model_quant.tflite", 'wb') as f:
        f.write(convert_to_tflite(quant_model))

