from prediction_model.model_constants import *
from prediction_model.quantizable_model import QuantModelWrapper
import tensorflow as tf
from compression.huffman import get_bits_from_file, extract_data_bits, decode_first_token_in_stream
from compression.compression_utils import get_quant_model_probs, deserialize_id_vocab
from compression.compression_constants import *


def decompress(source_dir, target_path, model_parameters=None):
    if not model_parameters:
        model_parameters = ModelParameters.deserialize(f"{source_dir}/{model_parameters_filename}")
    vocab, unknown = deserialize_id_vocab(f"{source_dir}/{vocab_filename}")
    interpreter = tf.lite.Interpreter(model_path=f"{source_dir}/{model_filename}")
    one_step_model = QuantModelWrapper(interpreter, vocab)

    input_bits = get_bits_from_file(f"{source_dir}/{data_filename}")
    input_data = extract_data_bits(input_bits)
    states = (model_parameters.default_init_states, model_parameters.default_init_carry)
    first_char = chr(int(input_data[:8], 2))
    next_char = first_char
    output_tokens = [first_char]
    start_index = 8
    while len(input_data) - start_index > 0:
        list_of_probs, states = get_quant_model_probs(one_step_model, next_char, states)
        decoded_token, start_index = decode_first_token_in_stream(list_of_probs, input_data, start_index, unknown_tokens=unknown)
        next_char = decoded_token
        output_tokens.append(decoded_token)
    output_text = "".join(output_tokens)
    output_file = open(target_path, "w")
    output_file.write(output_text)
    output_file.close()




if __name__ == '__main__':
    decompress("D:\\asaf\\יב\\compression learning\\data and stuff\\test data\\compressed_bible", f"D:\\asaf\\יב\\compression learning\\data and stuff\\test data\\bible.txt")
