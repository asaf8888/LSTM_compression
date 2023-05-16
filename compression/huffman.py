from queue import PriorityQueue
from collections import deque
import heapq
from bidict import bidict
from dataclasses import dataclass, field
from typing import Any
from collections import Counter


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def __str__(self):
        string = " "
        if self.data is not None:
            string += self.data
        if self.left is not None:
            string += str(self.left)
        if self.right is not None:
            string += str(self.right)
        return string


def create_coding_tree(list_of_probs, unknown_tokens=None):
    single_letter_q = []
    complex_q = deque()
    if unknown_tokens:
        tokens, prob = unknown_tokens
        single_token_prob = prob/len(tokens)
        list_of_probs.extend([(token, single_token_prob) for token in tokens])
    for token, prob in list_of_probs:
        heapq.heappush(single_letter_q, (PrioritizedItem(prob, Node(token))))

    while len(single_letter_q) + len(complex_q) > 1:
        lowest = get_lowest_of_2q(single_letter_q, complex_q)
        sec_lowest = get_lowest_of_2q(single_letter_q, complex_q)
        new_node = Node(None)
        new_node.left = lowest.item
        new_node.right = sec_lowest.item
        complex_q.append(PrioritizedItem(lowest.priority + sec_lowest.priority, new_node))

    return single_letter_q.pop().item if single_letter_q else complex_q.popleft().item


def get_lowest_of_2q(some_heapq, some_deque):
    if not some_deque:
        return heapq.heappop(some_heapq)
    elif not some_heapq:
        return some_deque.popleft()
    else:
        return heapq.heappop(some_heapq) if some_heapq[0] < some_deque[0] else some_deque.popleft()


def create_coding(list_of_probs, unknown_tokens=None):
    tree = create_coding_tree(list_of_probs, unknown_tokens)
    encoding_dict = create_coding_from_tree(tree)
    return bidict(encoding_dict)


def create_coding_from_tree(head):
    tree_search_stack = [(head, "")]
    encoding_dict = {}
    while tree_search_stack:
        head_and_encoding = tree_search_stack.pop()
        if head_and_encoding[0].data is not None:
            encoding_dict[head_and_encoding[1]] = head_and_encoding[0].data
        if head_and_encoding[0].left is not None:
            tree_search_stack.append((head_and_encoding[0].left, head_and_encoding[1] + "0"))
        if head_and_encoding[0].right is not None:
            tree_search_stack.append((head_and_encoding[0].right, head_and_encoding[1] + "1"))
    return encoding_dict


def encode_token(list_of_probs, token):
    coding_bidict = create_coding(list_of_probs)
    return coding_bidict.inverse.get(token)


def decode_first_token_in_stream(list_of_probs, string, start_index):
    coding_tree = create_coding_tree(list_of_probs)
    for bit in range(start_index, len(string)):
        coding_tree = coding_tree.left if string[bit] == '0' else coding_tree.right
        if coding_tree.data is not None:
            return coding_tree.data, bit+1


def encode_file(filename):
    input_file = open(filename, 'r')
    input_string = input_file.read()
    list_of_probs = list(dict(Counter(input_string)))
    encoded_list = [encode_token(list_of_probs, token) for token in input_string]
    encoded_output = "".join(encoded_list)
    data_in_bytes = fit_data_to_bytes(encoded_output)
    compressed_file = open(f"compressed_{filename}", "wb")
    compressed_file.write(data_in_bytes)
    compressed_file.close()


def fit_data_to_bytes(encoded_data_in_bits):
    last_byte_size = bin(((len(encoded_data_in_bits) + 4) % 8)).replace("0b", "")
    last_byte_size = last_byte_size.rjust(3, '0')
    data = '1' + last_byte_size + encoded_data_in_bits
    data = data.ljust(len(data) + 8 - (len(data) % 8), '0')
    data_in_bytes = int(data, 2).to_bytes(len(data) // 8, byteorder='big')
    return data_in_bytes


def decode_file(filename, list_of_probs):
    input_bits = get_bits_from_file(filename)
    input_data = extract_data_bits(input_bits)
    output_tokens = []
    start_index = 0
    while len(input_data) - start_index > 0:
        decoded_token, start_index = decode_first_token_in_stream(list_of_probs, input_data, start_index)
        output_tokens.append(decoded_token)
    output_text = "".join(output_tokens)
    return output_text


def get_bits_from_file(filename):
    input_file = open(filename, 'rb')
    input_string = input_file.read()
    input_bits = bin(int.from_bytes(input_string, byteorder='big')).replace("0b", "")
    return input_bits


def extract_data_bits(input_bits):
    last_byte_size = int(input_bits[1:4], 2)
    if last_byte_size != 0:
        input_bits = input_bits[4:last_byte_size-8]
    else:
        input_bits = input_bits[4:len(input_bits)]
    return input_bits


