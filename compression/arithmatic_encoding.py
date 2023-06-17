import time


from asaf_compression.compression.precise_fraction import PreciseFraction
def arithmetically_encode(list_of_probs, token, curr_range, unknown_tokens=None):
    start_idx = curr_range[0]
    range_factor = curr_range[1] - curr_range[0]
    if unknown_tokens:
        tokens, prob = unknown_tokens
        single_token_prob = prob / len(tokens)
        list_of_probs.extend([(token, single_token_prob) for token in tokens])
    # list_of_probs = map(lambda x: (x[0], PreciseFraction(*x[1].as_integer_ratio())), sorted(list_of_probs, key=lambda x: x[1], reverse=True))
    list_of_probs = sorted(list_of_probs, key=lambda x: x[1], reverse=True)
    for prob in list_of_probs:
        if prob[0] == token:
            end_index = start_idx + (prob[1] * range_factor)
            break
        start_idx += prob[1] * range_factor
    return (start_idx, end_index)


def expend_range_encode(curr_range):
    if(curr_range[1] < 0.5):
        return (curr_range[0] * 2, curr_range[1] * 2), '0'
    elif((curr_range[0] > 0.25) & (curr_range[1] < 0.75)):
        return ((curr_range[0] * 2) - 0.5, (curr_range[1] * 2) - 0.5), '2'
    elif(curr_range[0] > 0.5):
        return ((curr_range[0] * 2) - 1, (curr_range[1] * 2) - 1), '1'
    else:
        return curr_range, None
# def expend_range(curr_range):
#     if(curr_range[1] < PreciseFraction(1,2)):
#         return (curr_range[0] * PreciseFraction(2,1), curr_range[1] * PreciseFraction(2,1)), '0'
#     elif((curr_range[0] > PreciseFraction(1,4)) & (curr_range[1] < PreciseFraction(3,4))):
#         return ((curr_range[0] * PreciseFraction(2,1)) - PreciseFraction(1,2), (curr_range[1] * PreciseFraction(2,1) - PreciseFraction(1,2))), '2'
#     elif(curr_range[0] > PreciseFraction(1,2)):
#         return ((curr_range[0] * PreciseFraction(2,1)) - PreciseFraction(1,1), (curr_range[1] * PreciseFraction(2,1)) - PreciseFraction(1,1)), '1'
#     else:
#         return curr_range, None


def get_number_in_range(target_range):
    curr_range = (PreciseFraction(0,1), PreciseFraction(1,1))
    target = (target_range[0] + target_range[1]) * PreciseFraction(1, 2)
    result = []

    while(not((curr_range[0] > target_range[0]) & (curr_range[0] < target_range[1]))):
        middle = (curr_range[0] + curr_range[1]) * PreciseFraction(1, 2)
        if middle > target:
            result.append('0')
            curr_range =  (curr_range[0], middle)
        else:
            result.append('1')
            curr_range = (middle, curr_range[1])

    return ''.join(result)

# def decode_token(list_of_probs, number, curr_range, unknown_tokens=None):
#     start_idx = curr_range[0]
#     range_factor = curr_range[1] - curr_range[0]
#     if unknown_tokens:
#         tokens, prob = unknown_tokens
#         single_token_prob = prob / len(tokens)
#         list_of_probs.extend([(token, single_token_prob) for token in tokens])
#     list_of_probs = map(lambda x: (x[0], PreciseFraction(*x[1].as_integer_ratio())),
#                         sorted(list_of_probs, key=lambda x: x[1], reverse=True))
#     for prob in list_of_probs:
#         end_idx = start_idx + prob[1] * range_factor
#         if (number > start_idx) & (number < end_idx):
#             return prob[0], (start_idx, end_idx)
#         start_idx = end_idx


def decode_token(list_of_probs, curr_range, bit_stream, curr_target_range, unknown_tokens=None):
    range_factor = curr_target_range[1] - curr_target_range[0]
    if unknown_tokens:
        tokens, prob = unknown_tokens
        single_token_prob = prob / len(tokens)
        list_of_probs.extend([(token, single_token_prob) for token in tokens])
    list_of_probs = sorted(list_of_probs, key=lambda x: x[1], reverse=True)#map(lambda x: (x[0], PreciseFraction(*x[1].as_integer_ratio())),sorted(list_of_probs, key=lambda x: x[1], reverse=True))
    while True:
        start_idx = curr_target_range[0]
        for prob in list_of_probs:
            end_idx = start_idx + prob[1] * range_factor
            if (curr_range[0] >= start_idx):
                if  (curr_range[1] <= end_idx):
                    return prob[0], curr_range, bit_stream, (start_idx, end_idx)
            # elif curr_range[1] >= start_idx:
            #     break
            start_idx = end_idx
        if bit_stream:
            bit = int(bit_stream.popleft())
        else:
            bit = 0
        half = (curr_range[1] - curr_range[0])/2
        curr_range = (curr_range[0] + (bit * half), curr_range[1] - half + (bit * half))

def expend_range_decode(curr_range, curr_target_range):
    if(curr_target_range[1] < 0.5):
        return (curr_range[0] * 2, curr_range[1] * 2), (curr_target_range[0] * 2, curr_target_range[1] * 2), True
    elif((curr_target_range[0] > 0.25) & (curr_target_range[1] < 0.75)):
        return ((curr_range[0] * 2) - 0.5, (curr_range[1] * 2) - 0.5), ((curr_target_range[0] * 2) - 0.5, (curr_target_range[1] * 2) - 0.5), True
    elif(curr_target_range[0] > 0.5):
        return ((curr_range[0] * 2) - 1, (curr_range[1] * 2) - 1), ((curr_target_range[0] * 2) - 1, (curr_target_range[1] * 2) - 1), True
    else:
        return curr_range, curr_target_range, False

