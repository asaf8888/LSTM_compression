from prediction_model.model_constants import *
from prediction_model.my_model import MyModel
from compression.compression_constants import general_model_path
import re
import numpy

model = MyModel(vocab_size, embedding_dim, rnn_units)

model.load_weights(tf.train.latest_checkpoint(general_model_path))

the_bible_raw = open('../test data/bible.txt', 'r')
the_bible = the_bible_raw.readlines()
the_bible_raw.close()

the_bible = [x[re.search("\d+:\d+ *.", x).end(): ] for x in the_bible]
the_bible = [list(x) for x in the_bible]
the_bible = list(numpy.concatenate(the_bible).flat)
the_bible = list(map(lambda x: x.replace('\n', ' '), the_bible))

ids_dataset = tf.data.Dataset.from_tensor_slices(ids_from_chars(the_bible))
sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
dataset = sequences.map(lambda x: (x[:-1], x[1:]))

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics='accuracy')
model.evaluate(dataset)