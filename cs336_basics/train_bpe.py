import os
from tqdm import tqdm
from .pretokenization_example import get_pre_token_counts

NUM_BYTE_CHARS = 256

class TokenNode:
    def __init__(self, token: bytes | None = None, multiplier: int | None = None):
        self.token: bytes | None = token
        self.multiplier: int | None = multiplier
        self.prev: TokenNode | None = None
        self.next: TokenNode | None = None
    
    def __str__(self):
        if self.next:
            return f"({str(self.token)}, {self.multiplier}) -> {str(self.next)}"
        
        return f"({str(self.token)}, {self.multiplier})"


class TokenPairInfo:
    def __init__(self, occurence: TokenNode):
        self.occurences: list[TokenNode] = [occurence]
        self.num_occurences: int = occurence.multiplier
    
    def add_occurence(self, occurence: TokenNode):
        self.occurences.append(occurence)
        self.num_occurences += occurence.multiplier
    
    def decrease_occurence_count(self, occurence: TokenNode) -> int:
        # Only decrement the count no need to remove the occurence since it is marked as invalid
        old_num_occurences = self.num_occurences
        self.num_occurences -= occurence.multiplier

        return old_num_occurences

def merge_pair(pair: tuple[bytes, bytes]) -> bytes:
    '''Convenience function to merge a pair tuple bytes into a single bytes object'''

    return pair[0] + pair[1]

def merge_node(
        node: TokenNode,
        token_pairs: dict[tuple[bytes, bytes], TokenPairInfo]) -> dict[tuple[bytes, bytes], int]:
    
    changed_pair_counts = {}

    if node.prev.token:
        prev_pair = (node.prev.token, node.token)
        changed_pair_counts[prev_pair] = token_pairs[prev_pair].decrease_occurence_count(node.prev)

    if node.next.next.token:
        next_pair = (node.next.token, node.next.next.token)
        old_num_occurences = token_pairs[next_pair].decrease_occurence_count(node.next)
        
        if next_pair not in changed_pair_counts:
            changed_pair_counts[next_pair] = old_num_occurences
    
    # Merge the bytes of the node and its successor and re-link the linked list
    new_node = TokenNode(node.token + node.next.token, node.multiplier)
    new_node.prev = node.prev
    new_node.next = node.next.next
    node.prev.next = new_node
    node.next.next.prev = new_node
    # Disable the bytes of the nodes forming the pair.
    # This signals to the outside caller to skip the next node in the iteration should two identical pairs be adjacent.
    node.token = None
    node.next.token = None
    # Insert new pairs
    if new_node.prev.token:
        new_prev_pair = (new_node.prev.token, new_node.token)
        if new_prev_pair in token_pairs:
            token_pairs[new_prev_pair].add_occurence(new_node.prev)
        else:
            token_pairs[new_prev_pair] = TokenPairInfo(new_node.prev)
        changed_pair_counts[new_prev_pair] = None
        
    if new_node.next.token:
        new_pair = (new_node.token, new_node.next.token)
        if new_pair in token_pairs:
            token_pairs[new_pair].add_occurence(new_node)
        else:
            token_pairs[new_pair] = TokenPairInfo(new_node)
        changed_pair_counts[new_pair] = None
    
    return changed_pair_counts

def merge(
        token_pairs: dict[tuple[bytes, bytes], TokenPairInfo],
        cnt_to_pairs: dict[int, set[tuple[bytes, bytes]]],
        num_merges: int) -> list[tuple[bytes, bytes]]:
    # Create num_merges new tokens by merging the most frequent token pairs
    merge_sequence = []

    for _ in tqdm(range(num_merges)):
        # Exit if no more pairs can be merged
        if not cnt_to_pairs:
            break

        # Get the max occurence count
        max_cnt = max(cnt_to_pairs.keys())
        # Get the lexicographically greatest of all pairs sharing the max count
        max_pair = max(cnt_to_pairs[max_cnt])
        pair_info = token_pairs[max_pair]


        # Keep track of which pair counts need to be updated after merging a token
        changed_pair_counts = set()
        # Iterate over this pair's occurences
        for node in pair_info.occurences:
            if not node.token or (node.token, node.next.token) != max_pair:
                # The call to merge_node can disable a node in occurences or change the succeeding node's token.
                # If this is the case the node is skipped.
                continue

            new_changed_pair_counts = merge_node(node=node, token_pairs=token_pairs)

            for changed_pair, old_count in new_changed_pair_counts.items():
                # If this pair count change has not yet occured, remove it from the count dict
                if old_count and changed_pair not in changed_pair_counts:
                    if len(cnt_to_pairs[old_count]) == 1:
                        cnt_to_pairs.pop(old_count)
                    else:
                        cnt_to_pairs[old_count].remove(changed_pair)

            changed_pair_counts = changed_pair_counts.union(new_changed_pair_counts.keys())
        
        # Adjust the pair counts
        for changed_pair in changed_pair_counts:
            changed_pair_info = token_pairs[changed_pair]

            if changed_pair_info.num_occurences == 0:
                token_pairs.pop(changed_pair)
            elif changed_pair_info.num_occurences in cnt_to_pairs:
                cnt_to_pairs[changed_pair_info.num_occurences].add(changed_pair)
            else:
                cnt_to_pairs[changed_pair_info.num_occurences] = {changed_pair}
        
        # Remove pair and add to completed merges
        removed_info = token_pairs.pop(max_pair)
        if len(cnt_to_pairs[removed_info.num_occurences]) == 1:
            cnt_to_pairs.pop(removed_info.num_occurences)
        else:
            cnt_to_pairs[removed_info.num_occurences].remove(max_pair)

        merge_sequence.append(max_pair)
    
    return merge_sequence

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    pre_tokens = get_pre_token_counts(input_path, special_tokens, num_processes)

    # Build a linked list of tokens
    token_linked_list_start = TokenNode()
    current_node = token_linked_list_start

    for pre_token, count in pre_tokens.items():
        for byte_int in pre_token.encode('utf-8'):
            new_node = TokenNode(bytes([byte_int]), count)
            current_node.next = new_node
            new_node.prev = current_node
            current_node = new_node
        
        # Append separator between pre-tokens
        separator_node = TokenNode()
        current_node.next = separator_node
        separator_node.prev = current_node
        current_node = separator_node

    # Safe memory
    del pre_tokens

    # Let's make the start of the list a valid token
    token_linked_list_start = token_linked_list_start.next
    
    # Build a dict of token pairs to their occurences
    token_pairs: dict[tuple[bytes, bytes], TokenPairInfo] = {}
    current_node = token_linked_list_start
    
    while current_node:
        if current_node.next.token:
            # If the next token is not a separator insert the pair
            pair = (current_node.token, current_node.next.token)

            if pair in token_pairs:
                token_pairs[pair].add_occurence(current_node)
            else:
                token_pairs[pair] = TokenPairInfo(current_node)
            
            # Advance one node
            current_node = current_node.next
        else:
            # Advance two nodes to skip separator tokens
            current_node = current_node.next.next
    
    # Build a dict from counts to pairs
    cnt_to_pairs: dict[int, set[tuple[bytes, bytes]]] = {}
    current_node = token_linked_list_start

    for pair, pair_info in token_pairs.items():
        if pair_info.num_occurences in cnt_to_pairs:
            cnt_to_pairs[pair_info.num_occurences].add(pair)
        else:
            cnt_to_pairs[pair_info.num_occurences] = {pair}
    
    num_merges = vocab_size - len(special_tokens) - NUM_BYTE_CHARS

    merge_sequence = merge(token_pairs=token_pairs, cnt_to_pairs=cnt_to_pairs, num_merges=num_merges)

    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    num_tokens = len(vocab)
    vocab |= {num_tokens + i: bytes([i]) for i in range(NUM_BYTE_CHARS)}
    num_tokens = len(vocab)
    
    for i, pair in enumerate(merge_sequence):
        vocab[num_tokens + i] = merge_pair(pair)

    return vocab, merge_sequence
