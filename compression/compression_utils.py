from asaf_compression.compression.huffman import encode_token, fit_data_to_bytes
from asaf_compression.compression.compression_constants import unknown_character_token
from asaf_compression.compression.precise_fraction import PreciseFraction
from asaf_compression.compression.arithmatic_encoding import arithmetically_encode, get_number_in_range, expend_range_encode
import numpy as np

def get_quant_model_probs(quant_one_step_model, input_token, states):
    probs, states = quant_one_step_model.get_probabilty_weights(input_token, states)
    probs = probs.tolist()
    correctly_ordered_vocabulary = list(quant_one_step_model.vocab.keys())
    return tuple(zip(list(correctly_ordered_vocabulary), probs)), states


def compress_text_arithmetic(text, quantOneStep, model_parameters, unknown_tokens):
    states = model_parameters.default_init_states
    carry = model_parameters.default_init_carry

    first_char = text[0]
    next_char = first_char
    # resulting_range = (PreciseFraction(0,1), PreciseFraction(1,1))
    resulting_range = (0,1)
    counter = 0
    result = [format(ord(first_char), 'b').rjust(8, '0')]
    for token in text[1:len(text)]:
        print(f"{counter}/{len(text)-1}")
        list_of_probs, (states, carry) = get_quant_model_probs(quantOneStep, next_char, (states, carry))

        next_char = token
        resulting_range = arithmetically_encode(list_of_probs, token, resulting_range, unknown_tokens)
        resulting_range, bit = expend_range_encode(resulting_range)
        while bit:
            if (result[-1] == '2') & (bit != '2'):
                num_of_twos = 1
                while result[-num_of_twos-1] == '2':
                    num_of_twos += 1
                result[-num_of_twos] = bit
                for i in range(num_of_twos-1, 0, -1):
                    result[-i] = str(1-int(bit))
                result.append(str(1-int(bit)))
            else:
                result.append(bit)
            resulting_range, bit = expend_range_encode(resulting_range)
        counter +=1

    while result[-1] == '2':
        result.pop()
    result.append('1')
    result = "".join(result)

    data_in_bytes = fit_data_to_bytes(result)
    return data_in_bytes


def compress_text_huffman(text, quantOneStep, model_parameters, unknown_tokens):
    states = model_parameters.default_init_states
    carry = model_parameters.default_init_carry

    first_char = text[0]
    next_char = first_char
    result = [format(ord(first_char), 'b').rjust(8, '0')]
    for token in text[1:len(text)]:
        list_of_probs, (states, carry) = get_quant_model_probs(quantOneStep, next_char, (states, carry))

        next_char = token
        result.append(encode_token(list_of_probs, token, unknown_tokens))

    encoded_output = "".join(result)
    data_in_bytes = fit_data_to_bytes(encoded_output)
    return data_in_bytes


def serialize_id_vocab(filepath, vocab, unknown=None):
    vocab_file = open(filepath, "w")
    for token in vocab.keys():
        vocab_file.write(token)
    if unknown:
        for unknown_token in unknown:
            vocab_file.write(unknown_token)
    vocab_file.close()


def deserialize_id_vocab(filepath):
    vocab_file = open(filepath, "r")
    text = vocab_file.read()
    split_text = text.split(unknown_character_token)
    vocab_text = split_text[0]
    unknown = None
    if len(split_text) > 1:
        unknown = list(split_text[1])
    vocab = {vocab_text[i]: i for i in range(len(vocab_text))}
    vocab[unknown_character_token] = len(vocab)
    vocab_file.close()
    return vocab, unknown
