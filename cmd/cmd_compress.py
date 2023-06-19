import asaf_compression.compression.full_compression as fc
from asaf_compression.prediction_model.model_constants import ModelParameters
import sys
import os
filepath = sys.argv[1]
target_dir, filename = os.path.split(filepath)
compressed_filename = f"compressed_{os.path.splitext(filename)[0]}"
fc.compress(f"{target_dir}/{filename}", f"{target_dir}/{compressed_filename}", ModelParameters(50,100))