from prediction_model.my_model import MyModel
from compression.compression_constants import unknown_character_token
from collections import Counter
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
    masked_text = [token if token in vocab.keys() else unknown_character_token for token in text]
    id_vocab[unknown_character_token] = len(vocab)
    return masked_text, id_vocab, unknown


def prepare_training_dataset(text, vocab, batch_size, seq_lentgh=100):
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


def get_trained_model(text, model_parameters, batch_size, epochs, unknown_token_cutoff):
    masked_text, vocab, unknown = get_vocabulary_and_mask(text, unknown_token_cutoff)
    dataset = prepare_training_dataset(masked_text, vocab, batch_size=batch_size)
    vocab_size = len(vocab)
    model = MyModel(vocab_size=vocab_size, embedding_dim=model_parameters.embedding_dim, rnn_units=model_parameters.rnn_units)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer='adam', loss=loss, metrics='accuracy')
    model.fit(dataset, epochs=epochs)
    return model, (vocab, unknown)


