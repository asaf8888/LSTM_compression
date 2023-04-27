from constants import *
from model import MyModel, OneStep
import time
from huffman import encode_token, fit_data_to_bytes

model = MyModel(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
one_step_model = OneStep(model, vocab)

start = time.time()
states = None
result = []

input_file = open("bible.txt", 'r')
input_string = input_file.read()

next_char = tf.constant([input_string[0]])
i = 0
for token in input_string[1:len(input_string)]:
    i += 1
    probs, states = one_step_model.get_probs(next_char, states=states)
    probs = probs[0].numpy().tolist()
    correctly_ordered_vocabulary = chars_from_ids.get_vocabulary()
    list_of_probs = zip(list(correctly_ordered_vocabulary), probs)
    result.append(encode_token(list_of_probs, token))
    print(i, '/', len(input_string))

encoded_output = "".join(result)
data_in_bytes = fit_data_to_bytes(encoded_output)
compressed_file = open(f"compressed_bible", "wb")
compressed_file.write(data_in_bytes)
compressed_file.close()

end = time.time()
print('\nRun time:', end - start)
