EPOCHS = 1

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 32

import numpy as np
rng = np.random.default_rng(1)
default_init_states = rng.random((1, rnn_units), dtype=np.float32)
default_init_carry = rng.random((1, rnn_units), dtype=np.float32)


