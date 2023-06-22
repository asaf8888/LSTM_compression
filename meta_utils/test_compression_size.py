from asaf_compression.compression.full_compression import compress
from asaf_compression.prediction_model.model_constants import ModelParameters
from asaf_compression.compression.compression_constants import *
import os
import csv

if __name__ == '__main__':
    target_dir = f"D:\\asaf\\twelve grade\\compression learning\\data and stuff\\test data\\compressed_not_bible"
    output = []
    for rnn_units in [100]:
        for embedding_dim in [50]:

            compress("D:\\asaf\\twelve grade\\compression learning\\data and stuff\\test data\\not_bible.txt",
                     target_dir,
                     ModelParameters(embedding_dim, rnn_units), batch_size=1)
            model_size = os.stat(f"{target_dir}/{model_filename}").st_size
            data_size = os.stat(f"{target_dir}/{data_filename}").st_size
            output.append([rnn_units, embedding_dim, model_size, data_size])
    with open('../test_results.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(["rnn_units", "embedding_dim", "model_size", "data_size"])
        write.writerows(output)
