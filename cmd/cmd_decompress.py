import compression.my_consugation as consug
import sys
import os
filepath = sys.argv[1]
target_dir, filename = os.path.split(filepath)
decompressed_filename = filename.replace("compressed_", "") + ".txt"
consug.decompress_arithmatic(f"{target_dir}/{filename}", f"{target_dir}/{decompressed_filename}")