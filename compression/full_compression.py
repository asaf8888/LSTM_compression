from prediction_model.quantizable_model import QuantOneStep, get_quantizable_model, convert_to_tflite
import time
from prediction_model.training_utils import get_trained_model
from compression.compression_utils import compress_text, serialize_id_vocab, deserialize_id_vocab
from compression.compression_constants import *
from prediction_model.model_constants import ModelParameters
import tensorflow as tf
import os

def compress(filepath, target_dir, model_parameters, train_target=None):
    start = time.time()
    input_file = open(filepath, 'r')
    input_string = input_file.read()
    input_file.close()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if train_target is None:
        model, (vocab, unknown) = get_trained_model(input_string, model_parameters)
    else:
        model, (vocab, unknown) = get_trained_model(train_target, model_parameters)
    serialize_id_vocab(f"{target_dir}/{vocab_filename}", vocab, unknown)
    quantizable_model = get_quantizable_model(model, model_parameters)
    quant_model = convert_to_tflite(quantizable_model)

    with open(f"{target_dir}/{model_filename}", 'wb') as f:
        f.write(quant_model)

    interpreter = tf.lite.Interpreter(model_content=quant_model)
    quant_one_step = QuantOneStep(interpreter, vocab)
    data_in_bytes = compress_text(input_string, quant_one_step, model_parameters, unknown)

    compressed_file = open(f"{target_dir}/{data_filename}", "wb")
    compressed_file.write(data_in_bytes)
    compressed_file.close()

    model_parameters.serialize(f"{target_dir}/{model_parameters_filename}")

    end = time.time()
    print('\nRun time:', end - start)


if __name__ == '__main__':
    # input_file = open("../test data/bible.txt", 'r')
    # input_string = input_file.read()
    # input_file.close()
    compress("D:\\asaf\\יב\\compression learning\\data and stuff\\test data\\bible.txt", f"D:\\asaf\\יב\\compression learning\\data and stuff\\test data\\compressed_not_bible", ModelParameters(256, 32))
