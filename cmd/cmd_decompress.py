import asaf_compression.compression.my_consugation as consug
from asaf_compression.prediction_model.model_constants import ModelParameters
import sys
import os
filepath = sys.argv[1]
target_dir, filename = os.path.split(filepath)
decompressed_filename = filename.replace("compressed_", "") + ".txt"
consug.decompress(f"{target_dir}/{filename}", f"{target_dir}/{decompressed_filename}")