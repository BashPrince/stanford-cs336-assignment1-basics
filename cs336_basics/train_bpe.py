import os
import regex
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
    
    def remove_occurence(self, occurence: TokenNode) -> int:
        self.occurences.remove(occurence)
        old_num_occurences = self.num_occurences
        self.num_occurences -= occurence.multiplier

        return old_num_occurences


def pre_tokenize(corpus: str, special_tokens: list[str]) -> dict[str, int]:
    special_tokens = [regex.escape(t) for t in special_tokens]
    # Split corpus along special tokens
    corpus_splits = regex.split("|".join(special_tokens), corpus)

    pre_tokens = {}

    for split in corpus_splits:
        match_itr = regex.finditer(PAT, split)
        
        for match_obj in match_itr:
            if match_obj:
                match_group = match_obj.group(0)
                if match_group in pre_tokens.keys():
                    pre_tokens[match_group] += 1
                else:
                    pre_tokens[match_group] = 1
    
    return pre_tokens

def merge_pair(pair: tuple[bytes, bytes]) -> bytes:
    '''Convenience function to merge a pair tuple bytes into a single bytes object'''

    return pair[0] + pair[1]

def merge_node(
        node: TokenNode,
        token_pairs: dict[tuple[bytes, bytes], TokenPairInfo]) -> dict[tuple[bytes, bytes], int]:
    
    changed_pair_counts = {}

    if node.prev.token:
        prev_pair = (node.prev.token, node.token)
        changed_pair_counts[prev_pair] = token_pairs[prev_pair].remove_occurence(node.prev)

    if node.next.next.token:
        next_pair = (node.next.token, node.next.next.token)
        old_num_occurences = token_pairs[next_pair].remove_occurence(node.next)
        
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

def get_pre_token_by_occurence(occurence: TokenNode) -> tuple[bytes]:
    while occurence.prev.token is not None:
        occurence = occurence.prev
    
    pre_token = [occurence.token]

    while occurence.next.token is not None:
        occurence = occurence.next
        pre_token.append(occurence.token)
    
    return tuple(pre_token)

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
            if not node.token:
                # The call to merge_node can disable a node in occurences. If this is the case the node is skipped.
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
    
    pool = Pool(num_processes)

    with open(input_path, "rb") as f:
        # Prepare the chunks
        boundaries = find_chunk_boundaries(f, pool._processes, "<|endoftext|>".encode("utf-8"))
        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
        
        # Run pre-tokenization on all chunks and store the chunk counts for each pre-token
        chunk_pre_tokens_list = pool.starmap(pre_tokenize, zip(chunks, repeat(special_tokens)))
        
        # Merge all chunk pre-tokens and counts into one dict
        pre_tokens = chunk_pre_tokens_list[0]

        if len(chunk_pre_tokens_list) > 1:
            for chunk_pre_tokens in chunk_pre_tokens_list[1:]:
                for pre_token, count in chunk_pre_tokens.items():
                    if pre_token in pre_tokens:
                        pre_tokens[pre_token] += count
                    else:
                        pre_tokens[pre_token] = count

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
    
def merge_naive(pre_tokens: list[tuple[list[bytes], int]], num_merges: int):
    """Iterator that returns the next merge and the token pair counts after each step"""

    # Perform merges
    for _ in range(num_merges):
        pairs = {}
        max_pair = None
        max_count = 0
        pre_tokens_merged = []

        # Find the most often occuring pair
        for pt, pt_count in pre_tokens:
            # Iterate over pre-token tokens pairwise
            for t1, t2 in zip(pt[:-1], pt[1:]):
                pair = (t1, t2)
                pair_count = pairs.get(pair, 0)
                new_count = pair_count + pt_count
                pairs[pair] = new_count

                if (new_count > max_count or max_pair is None) or (new_count == max_count and pair > max_pair):
                    max_count = new_count
                    max_pair = pair
        
        # If no new max pair was found finish
        if max_pair is None:
            break

        yield max_pair, pairs
        
        # Merge the most often occuring pair
        for pt, pt_count in pre_tokens:
            pre_token_merged = []

            skip = False
            for prev_t, t1, t2, next_t in zip(([None] + pt)[:-2], pt[:-1], pt[1:], (pt + [None])[2:]):
                if skip:
                    skip = False
                    if next_t is None:
                        pre_token_merged.append(t2)

                    continue

                pair = (t1, t2)
                merged_pair = merge_pair(pair)

                if pair == max_pair:
                    if prev_t:
                        prev_pair = (prev_t, t1)
                        pairs[prev_pair] -= pt_count

                        if pairs[prev_pair] == 0:
                            pairs.pop(prev_pair)

                        new_pair = (prev_t, merged_pair)

                        if new_pair in pairs:
                            pairs[new_pair] += pt_count
                        else:
                            pairs[new_pair] = pt_count

                    if next_t:
                        next_pair = (t2, next_t)
                        pairs[next_pair] -= pt_count

                        if pairs[next_pair] == 0:
                            pairs.pop(next_pair)

                        new_pair = (merged_pair, next_t)

                        if new_pair in pairs:
                            pairs[new_pair] += pt_count
                        else:
                            pairs[new_pair] = pt_count
                    
                    pre_token_merged.append(merged_pair)
                    skip = True
                else:
                    pre_token_merged.append(t1)

                    if next_t is None:
                        pre_token_merged.append(t2)

            if len(pre_token_merged) > 1:
                pre_tokens_merged.append((pre_token_merged, pt_count))
        
        pre_tokens = pre_tokens_merged

def train_bpe_naive(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    pool = Pool(num_processes)

    with open(input_path, "rb") as f:
        # Prepare the chunks
        boundaries = find_chunk_boundaries(f, pool._processes, "<|endoftext|>".encode("utf-8"))
        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
        
        # Run pre-tokenization on all chunks and store the chunk counts for each pre-token
        chunk_pre_tokens_list = pool.starmap(pre_tokenize, zip(chunks, repeat(special_tokens)))
        
        # Merge all chunk pre-tokens and counts into one dict
        pre_tokens = chunk_pre_tokens_list[0]

        if len(chunk_pre_tokens_list) > 1:
            for chunk_pre_tokens in chunk_pre_tokens_list[1:]:
                for pre_token, count in chunk_pre_tokens.items():
                    if pre_token in pre_tokens:
                        pre_tokens[pre_token] += count
                    else:
                        pre_tokens[pre_token] = count
        
        pre_tokens = [(pt.encode("utf-8"), count) for pt, count in pre_tokens.items()]
        pre_tokens = [([pt[i:i+1] for i in range(len(pt))], count) for pt, count in pre_tokens if len(pt) > 1]

    merge_itr = merge_naive(pre_tokens=pre_tokens, num_merges=vocab_size - len(special_tokens) - NUM_BYTE_CHARS)
    merge_sequence = [m for m, _ in merge_itr]
            
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    num_tokens = len(vocab)
    vocab |= {num_tokens + i: bytes([i]) for i in range(NUM_BYTE_CHARS)}
    num_tokens = len(vocab)
    
    for i, pair in enumerate(merge_sequence):
        vocab[num_tokens + i] = merge_pair(pair)

    return vocab, merge_sequence