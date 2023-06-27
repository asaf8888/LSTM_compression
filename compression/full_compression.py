from asaf_compression.prediction_model.quantizable_model import QuantModelWrapper, get_quantizable_model, convert_to_tflite
from asaf_compression.prediction_model.training_utils import get_trained_model
from asaf_compression.compression.compression_utils import compress_text_huffman, compress_text_arithmetic, serialize_id_vocab
from asaf_compression.compression.compression_constants import *
from asaf_compression.prediction_model.model_constants import ModelParameters
from asaf_compression.prediction_model.model_constants import *
import tensorflow as tf
import os
import json

def compress(filepath, target_dir, model_parameters, batch_size=default_batch_size, epochs=EPOCHS, unknown_token_cutoff=0, train_target=None):
    input_file = open(filepath, 'r')
    input_string = input_file.read()
    input_file.close()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if train_target is None:
        model, (vocab, unknown) = get_trained_model(input_string, model_parameters, batch_size=batch_size, epochs=epochs, unknown_token_cutoff=unknown_token_cutoff)
    else:
        model, (vocab, unknown) = get_trained_model(train_target, model_parameters, batch_size=batch_size, epochs=epochs, unknown_token_cutoff=unknown_token_cutoff)
    serialize_id_vocab(f"{target_dir}/{vocab_filename}", vocab, unknown)
    quantizable_model = get_quantizable_model(model, model_parameters)
    quant_model = convert_to_tflite(quantizable_model)

    file = open(f"{target_dir}/{model_filename}", "wb")
    file.write(quant_model)
    file.close()

    interpreter = tf.lite.Interpreter(model_content=quant_model)
    quant_one_step = QuantModelWrapper(interpreter, vocab)
    data_in_bytes = compress_text_arithmetic(input_string, quant_one_step, model_parameters, unknown)#compress_text_huffman(input_string, quant_one_step, model_parameters, unknown)

    compressed_file = open(f"{target_dir}/{data_filename}", "wb")
    compressed_file.write(data_in_bytes)
    compressed_file.close()

    model_parameters.serialize(f"{target_dir}/{model_parameters_filename}")
    meta_data = {"character count": len(input_string)}
    meta_data_json_representation = json.dumps(meta_data)
    file = open(f"{target_dir}/{meta_data_filename}", "w")
    file.write(meta_data_json_representation)
    file.close()


if __name__ == '__main__':
    compress("D:\\asaf\\twelve grade\\compression learning\\data and stuff\\test data\\not_bible.txt", f"D:\\asaf\\twelve grade\\compression learning\\data and stuff\\test data\\compressed_not_bible", ModelParameters(50, 100), batch_size=1, epochs=1)

