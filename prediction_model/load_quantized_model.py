from prediction_model.my_model import MyModel, OneStep
from prediction_model.quantizable_model import QuantOneStep
from prediction_model.model_constants import *
from compression.compression_utils import get_quant_model_probs, get_model_probs
from compression.compression_constants import *
import numpy as np

rng = np.random.default_rng()
states = rng.random((1, rnn_units), dtype=np.float32)
carry = rng.random((1, rnn_units), dtype=np.float32)
interpreter = tf.lite.Interpreter(model_path="../quantized_bible_model/model_quant.tflite")
one_step_model = QuantOneStep(interpreter, vocab)
probs, out_states = get_quant_model_probs(one_step_model, "a", (states, carry))
print(probs)
for i in range(5):
    input = max(probs, key=lambda x: x[1])[0]
    print(max(probs, key=lambda x: x[1]))
    probs, out_states = get_quant_model_probs(one_step_model, input, out_states)
#
model = MyModel(vocab_size, embedding_dim, rnn_units)
model.load_weights(tf.train.latest_checkpoint('../small_bible_model'))
one_step_model = OneStep(model, vocab)
probs, out_states = get_model_probs(one_step_model, ["a"], (states, carry))
for i in range(5):
    input = [max(probs, key=lambda x: x[1])[0]]
    print(max(probs, key=lambda x: x[1]))
    probs, out_states = get_model_probs(one_step_model, input, out_states)


