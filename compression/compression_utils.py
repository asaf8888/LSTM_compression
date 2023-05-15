import tensorflow as tf
def get_model_probs(one_step_model, input_token, states):
    probs, states = one_step_model.get_probs(input_token, states=states)
    probs = probs[0].numpy().tolist()
    correctly_ordered_vocabulary = one_step_model.chars_from_ids.get_vocabulary()
    return tuple(zip(list(correctly_ordered_vocabulary), probs)), states


def get_quant_model_probs(quant_one_step_model, input_token, states):
    probs, states = quant_one_step_model.get_probabilty_weights(input_token, states)
    probs = probs.tolist()
    correctly_ordered_vocabulary = quant_one_step_model.chars_from_ids.get_vocabulary()
    return tuple(zip(list(correctly_ordered_vocabulary), probs)), states
