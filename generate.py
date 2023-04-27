from constants import *
from model import MyModel, OneStep
import time

model = MyModel(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
one_step_model = OneStep(model, vocab)

start = time.time()
states = None
next_char = tf.constant(['G'])
result = [next_char]

for n in range(1):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
print('\nRun time:', end - start)
