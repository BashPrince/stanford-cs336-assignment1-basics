import pickle
import regex
from typing import Iterable, Iterator
from .pretokenization_example import PAT

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None):
        
        self.vocab = vocab
        self.inverted_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens
        self.special_token_regex = None

        if special_tokens:
            # Sort special tokens by length descending to handle overlapping tokens
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            special_tokens_sorted = [regex.escape(t) for t in special_tokens_sorted]
            self.special_token_regex = "|".join(special_tokens_sorted)
        
        # Build a dict of merges with their occurence order as values
        self.merges = {m: i for i, m in enumerate(merges)}

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None):
        
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)

        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)
        
        return cls(vocab, merges, special_tokens)
    
    def _find_next_merge(self, pre_token: tuple[bytes], start: int) -> tuple[tuple[bytes] | None, int | None]:
        '''Return the next applicable merge and its position from the start position of merges'''

        if len(pre_token) == 1:
            # Token is fully merged
            return None, None
        
        lowest_merge = None
        lowest_position = None

        # Iterate token pairs
        for t1, t2 in zip(pre_token[:-1], pre_token[1:]):
            pair = (t1, t2)

            # Check if the pair is a valid merge
            if pair in self.merges:
                position = self.merges[pair]

                # Skip if the merge lies before the start position
                if position < start:
                    continue

                # If this is the first merge or it occurs before the previously found lowest merge, save it
                if lowest_merge is None or position < lowest_position:
                    lowest_merge = pair
                    lowest_position = position
        
        return lowest_merge, lowest_position
    
    def _apply_merge(self, pre_token: tuple[bytes], merge: tuple[bytes]) -> tuple[bytes]:
        '''Apply all occurrences of the merge to the pre-token in order of appearance'''

        merge_applied_pre_token = []
        skip = False

        for t1, t2 in zip(pre_token[:-1], pre_token[1:]):
            if skip:
                skip = False
                continue

            if (t1, t2) == merge:
                # Append the merge and skip the next token
                merge_applied_pre_token.append(t1 + t2)
                skip = True
            else:
                # Append unmerged token
                merge_applied_pre_token.append(t1)
        
        if not skip:
            # The last pair did not get merged so we need to append the trailing bytes
            merge_applied_pre_token.append(pre_token[-1])
        
        return tuple(merge_applied_pre_token)

    def encode(self, text: str) -> list[int]:
        # Split the text into sequences of non-special token text and special tokens
        split_sequence = []
        special_sequence = []
        prev_match = None

        if self.special_token_regex:
            special_itr = regex.finditer(self.special_token_regex, text)

            for curr_match in special_itr:
                # Append the matched special token
                special_sequence.append(curr_match.group(0).encode("utf-8"))
                # Append the text from the end of the last match to the start of the current match
                prev_match_end = prev_match.end() if prev_match else 0
                split_sequence.append(text[prev_match_end:curr_match.start()])

                prev_match = curr_match
        

        # Append the trailing end split or the entire text if no match was found
        if not prev_match:
            split_sequence.append(text)
        elif prev_match.end() != len(text):
            split_sequence.append(text[prev_match.end():])


        # Convert the special tokens to ints
        special_sequence = [self.inverted_vocab[t] for t in special_sequence]


        # Build a sequence over splits of pre-token byte sequences
        pre_token_seq_seq = []

        for split in split_sequence:
            match_itr = regex.finditer(PAT, split)
            split_pre_tokens = []

            for match_obj in match_itr:
                if match_obj:
                    pre_token_bytes = match_obj.group(0).encode('utf-8')
                    pre_token_byte_tuple = tuple([bytes([b]) for b in pre_token_bytes])
                    split_pre_tokens.append(pre_token_byte_tuple)
            
            pre_token_seq_seq.append(split_pre_tokens)
            

        # Build a sequence over splits of merged pre-token byte sequences
        merged_pre_token_seq_seq = []

        for split in pre_token_seq_seq:
            split_merged_pre_tokens = []

            for pre_token in split:
                # Append single byte tokens and empty pre-tokens and continue with next pre-token
                if len(pre_token) <= 1:
                    split_merged_pre_tokens.append(pre_token)
                    continue

                merge_pos = 0

                while True:
                    next_merge, merge_pos = self._find_next_merge(pre_token, merge_pos)

                    if next_merge is None:
                        break

                    pre_token = self._apply_merge(pre_token, next_merge)
                
                split_merged_pre_tokens.append(pre_token)
            
            merged_pre_token_seq_seq.append(split_merged_pre_tokens)


        # Convert the sequence of merged pre-token byte sequences into a sequence of int sequences
        pre_token_int_seq_seq = []
        for split in merged_pre_token_seq_seq:
            split_pre_token_int_seqs = []

            for merged_pre_token in split:
                split_pre_token_int_seqs.append(tuple([self.inverted_vocab[t] for t in merged_pre_token]))

            pre_token_int_seq_seq.append(split_pre_token_int_seqs)
        

        # Fuse the special token and pre-token integer sequences
        int_seq = []
        if len(special_sequence) < len(pre_token_int_seq_seq):
            # If necessary append an invalid element for zipped iteration
            special_sequence.append(None)

        for split, special_int in zip(pre_token_int_seq_seq, special_sequence):
            # A split (can be empty) is followed by a special token
            for pre_token_ints in split:
                int_seq += [i for i in pre_token_ints]
            
            if special_int is not None:
                int_seq.append(special_int)
        
        return int_seq
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        def encode_iterator(iterable: Iterable[str]) -> Iterator[int]:
            for text in iterable:
                token_ints = self.encode(text)

                for i in token_ints:
                    yield i
        
        return encode_iterator(iterable)

    def decode(self, ids: list[int]) -> str:
        concat_bytes = b''

        for id in ids:
            concat_bytes += self.vocab[id]
        
        return concat_bytes.decode('utf-8', errors='replace')
