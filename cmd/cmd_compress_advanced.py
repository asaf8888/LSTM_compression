import compression.full_compression as fc
from prediction_model.model_constants import *
import sys
import os
filepath = sys.argv[1]
target_dir, filename = os.path.split(filepath)
compressed_filename = f"compressed_{os.path.splitext(filename)[0]}"
batch_size = int(input("what batch size do you want? \nwrite 0 for default value\n"))
embedding = int(input("how large should the embedding layer be? \nwrite 0 for default value\n"))
lstm_size = int(input("how large should the LSTM layer be? \nwrite 0 for default value\n"))
epochs = int(input("how many training epochs should run? \nwrite 0 for default value\n"))
unknown_charecter_cutoff = float(input("choose percent of tokens to be the minimum for the model to remember a token? \nwrite 0 for default value\n"))
if not batch_size:
    batch_size = default_batch_size
if not embedding:
    embedding = default_params.embedding_dim
if not lstm_size:
    lstm_size = default_params.rnn_units
if not epochs:
    epochs = EPOCHS

fc.compress(f"{target_dir}/{filename}", f"{target_dir}/{compressed_filename}", ModelParameters(embedding,lstm_size), epochs=epochs, batch_size=batch_size, unknown_token_cutoff=unknown_charecter_cutoff)