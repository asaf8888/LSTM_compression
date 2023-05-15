from prediction_model.my_model import MyModel
from prediction_model.quantizable_model import get_quantizable_model, convert_to_tflite
from prediction_model.model_constants import *

model = MyModel(vocab_size, embedding_dim, rnn_units)

model.load_weights(tf.train.latest_checkpoint("../large_bible_model"))
quant_model = get_quantizable_model(model)

with open("../quantized_bible_model/model_quant.tflite", 'wb') as f:
    f.write(convert_to_tflite(quant_model))