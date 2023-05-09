from my_constants import *
from my_model import MyModel, OneStep
import time
from huffman import get_bits_from_file, extract_data_bits, decode_first_token_in_stream
from compression.compression_utils import get_model_probs

model = MyModel(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
one_step_model = OneStep(model, vocab)

start = time.time()
states = None

input_bits = get_bits_from_file("not_compressed_bible")
input_data = extract_data_bits(input_bits)
first_char = chr(int(input_data[:8], 2))
next_char = tf.constant([first_char])
output_tokens = [first_char]
start_index = 8
while len(input_data) - start_index > 0:
    list_of_probs, states = get_model_probs(one_step_model, next_char, states)
    decoded_token, start_index = decode_first_token_in_stream(list_of_probs, input_data, start_index)
    next_char = tf.constant([decoded_token])
    output_tokens.append(decoded_token)
output_text = "".join(output_tokens)
print(output_text)
