from my_constants import *
from my_model import MyModel, OneStep
import time
from huffman import encode_token, fit_data_to_bytes
from compression.compression_utils import get_model_probs

model = MyModel(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
one_step_model = OneStep(model, vocab)

start = time.time()
states = None

input_file = open("not_bible.txt", 'r')
input_string = input_file.read()

first_char = input_string[0]
next_char = tf.constant([first_char])
result = [format(ord(first_char), 'b').rjust(8, '0')]
print(result)
for token in input_string[1:len(input_string)]:
    list_of_probs, states = get_model_probs(one_step_model, next_char, states)
    next_char = tf.constant([token])
    result.append(encode_token(list_of_probs, token))

encoded_output = "".join(result)
data_in_bytes = fit_data_to_bytes(encoded_output)
compressed_file = open(f"not_compressed_bible", "wb")
compressed_file.write(data_in_bytes)
compressed_file.close()

end = time.time()
print('\nRun time:', end - start)
