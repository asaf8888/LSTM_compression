import numpy as np
import json
EPOCHS = 80


class ModelParameters:

    def __init__(self, embedding_dim, rnn_units):
        self.embedding_dim = embedding_dim

        self.rnn_units = rnn_units

        rng = np.random.default_rng(1)
        self.default_init_states = rng.random((1, rnn_units), dtype=np.float32)
        self.default_init_carry = rng.random((1, rnn_units), dtype=np.float32)

    def serialize(self, filepath):
        dict_representation = {
            "rnn_units": self.rnn_units,
            "embedding_dim": self.embedding_dim
        }
        json_representation = json.dumps(dict_representation)
        file = open(filepath, "w")
        file.write(json_representation)
        file.close()

    @staticmethod
    def deserialize(filepath):
        file = open(filepath, "r")
        json_representation = file.read()
        file.close()
        dict_representation = json.loads(json_representation)
        return ModelParameters(dict_representation["embedding_dim"], dict_representation["rnn_units"])
