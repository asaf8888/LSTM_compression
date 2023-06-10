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

def get_number_in_range(target_range):
    curr_range = (PreciseFraction(0,1), PreciseFraction(1,1))
    target = (target_range[0] + target_range[1]) * PreciseFraction(1, 2)
    target.print_real()
    result = []

    while(not((curr_range[0] > target_range[0]) & (curr_range[0] < target_range[1]))):
        middle = (curr_range[0] + curr_range[1]) * PreciseFraction(1, 2)
        if middle > target:
            result.append('0')
            curr_range =  (curr_range[0], middle)
        else:
            result.append('1')
            curr_range = (middle, curr_range[1])
        curr_range[0].print_real()

    return ''.join(result)

def decode_token(list_of_probs, number, curr_range, unknown_tokens=None):
    start_idx = curr_range[0]
    range_factor = curr_range[1] - curr_range[0]
    if unknown_tokens:
        tokens, prob = unknown_tokens
        single_token_prob = prob / len(tokens)
        list_of_probs.extend([(token, single_token_prob) for token in tokens])
    list_of_probs = map(lambda x: (x[0], PreciseFraction(*x[1].as_integer_ratio())),
                        sorted(list_of_probs, key=lambda x: x[1], reverse=True))
    for prob in list_of_probs:
        end_idx = start_idx + prob[1] * range_factor
        if (number > start_idx) & (number < end_idx):
            return prob[0], (start_idx, end_idx)
        start_idx = end_idx

