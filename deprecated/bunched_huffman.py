from asaf_compression.compressionhuffman import create_coding
import math
import numpy as np
from matplotlib import pyplot as plt

def get_compression_ratio_coin_tosses(p, seq_len, clumping_size):
    # print(f"entropy per flip: {entropy_per_flip}")
    import random
    seq = ['1' if random.random() <= p else '0' for i in range(seq_len)]

    # print("".join(seq))
    # print(f"length: {length}")
    # print(f"entropy of sequence: {length*entropy_per_flip}")
    def get_list_of_probs(start, prob, list):
        if len(start) == clumping_size:
            list.append((start, prob))
        else:
            get_list_of_probs(start + "0", prob * (1 - p), list)
            get_list_of_probs(start + "1", prob * p, list)

    list_of_probs = []
    get_list_of_probs("", 1, list_of_probs)
    coding_bidict = create_coding(list_of_probs)
    clumped_seq = (seq[pos:pos + clumping_size] for pos in range(0, len(seq), clumping_size))
    output = "".join([coding_bidict.inverse.get("".join(token)) for token in clumped_seq])
    # print(output)
    # print(len(output))
    return len(output) / seq_len


if __name__ == '__main__':
    length = 2 * 3 * 2 * 5 * 7 * 2 * 3 * 11
    p = 0.8
    entropy_per_flip = -(p * math.log(p, 2) + (1 - p) * math.log((1 - p), 2))
    print(length)
    res = np.ndarray((12, 20))
    for i in range(1, 13):
        print(i)
        for j in range(20):
            res[i - 1, j] = get_compression_ratio_coin_tosses(p, length, i)
    avg = np.mean(res, 1)
    std = np.std(res, 1)
    print(std)
    plt.errorbar(range(1, 13), avg, std)
    plt.plot(range(1, 13), [entropy_per_flip for i in range(12)])
    plt.show()