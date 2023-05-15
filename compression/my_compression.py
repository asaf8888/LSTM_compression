from prediction_model.model_constants import *
from prediction_model.my_model import MyModel, OneStep
from prediction_model.quantizable_model import QuantOneStep
import time
from compression.huffman import encode_token, fit_data_to_bytes
from compression.compression_utils import get_model_probs, get_quant_model_probs
from compression.compression_constants import *
import numpy as np


def compress(filepath, interpreter, use_general_model=True):
    one_step_model = QuantOneStep(interpreter, vocab)

    rng = np.random.default_rng()
    states = rng.random((1, rnn_units), dtype=np.float32)
    carry = rng.random((1, rnn_units), dtype=np.float32)

    start = time.time()

    input_file = open(filepath, 'r')
    input_string = input_file.read()

    first_char = input_string[0]
    next_char = first_char
    result = [format(ord(first_char), 'b').rjust(8, '0')]
    for token in input_string[1:len(input_string)]:
        list_of_probs, (states, carry) = get_quant_model_probs(one_step_model, next_char, (states, carry))
        next_char = token
        result.append(encode_token(list_of_probs, token))

    encoded_output = "".join(result)
    data_in_bytes = fit_data_to_bytes(encoded_output)
    slash_idx = filepath.rfind('/')
    directory_path, filename = filepath[:slash_idx+1], filepath[slash_idx+1:-4]
    compressed_file = open(f"{directory_path}compressed_{filename}", "wb")
    compressed_file.write(data_in_bytes)
    compressed_file.close()

    end = time.time()
    print('\nRun time:', end - start)


if __name__ == '__main__':
    compress("../test data/bible.txt", tf.lite.Interpreter(model_path="../quantized_bible_model/model_quant.tflite"))
