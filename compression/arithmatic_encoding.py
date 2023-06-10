from precise_fraction import PreciseFraction
def arithmetically_encode(list_of_probs, token, curr_range, unknown_tokens=None):
    start_idx = curr_range[0]
    range_factor = curr_range[1] - curr_range[0]
    if unknown_tokens:
        tokens, prob = unknown_tokens
        single_token_prob = prob / len(tokens)
        list_of_probs.extend([(token, single_token_prob) for token in tokens])
    list_of_probs = map(lambda x: (x[0], PreciseFraction(*x[1].as_integer_ratio())), sorted(list_of_probs, key=lambda x: x[1], reverse=True))
    for prob in list_of_probs:
        if prob[0] == token:
            end_index = start_idx + (prob[1] * range_factor)
            break
        start_idx += prob[1] * range_factor
    return (start_idx, end_index)



if __name__ == '__main__':
    list_of_probs = (("a", 0.5), ("b", 0.2), ("c", 0.2), ("d", 0.1))
    list_of_probs2 = (("a", 0.1), ("b", 0.6), ("c", 0.2), ("d", 0.1))
    curr_range = (PreciseFraction(0,1), PreciseFraction(1,1))
    curr_range = arithmetically_encode(list_of_probs, "a", curr_range)
    curr_range = arithmetically_encode(list_of_probs2, "a", curr_range)
    curr_range = arithmetically_encode(list_of_probs, "b", curr_range)
    print(curr_range)
