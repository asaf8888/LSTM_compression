import compression.full_compression as fc
from prediction_model.model_constants import *
import sys
import os
filepath = sys.argv[1]
target_dir, filename = os.path.split(filepath)
compressed_filename = f"compressed_{os.path.splitext(filename)[0]}"
fc.compress(f"{target_dir}/{filename}", f"{target_dir}/{compressed_filename}", default_params, batch_size=default_batch_size)