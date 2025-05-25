import os
import regex
from itertools import repeat
from multiprocessing import Pool
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_CHUNK_BYTES = 1024 ** 3 # 1GB

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunk_pre_token_counts(corpus: str, special_tokens: list[str]) -> dict[str, int]:
    '''Get a dict of all occuring pre-tokens and their counts'''

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

def chunk_pre_token_list_iterator(
        input_path: str,
        num_processes: int):
    
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        num_chunks = file_size // MAX_CHUNK_BYTES + min(file_size % MAX_CHUNK_BYTES, 1)

        # Prepare the chunks
        boundaries = find_chunk_boundaries(f, num_chunks * num_processes, "<|endoftext|>".encode("utf-8"))
        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            if len(chunks) == num_processes:
                yield chunks
                chunks = []
        
        if chunks:
            yield chunks

def get_pre_token_counts(
        input_path: str,
        special_tokens: list[str],
        num_processes: int = None) -> dict[str, int]:
    
    pool = Pool(num_processes)
    chunk_itr = chunk_pre_token_list_iterator(input_path, pool._processes)
    pre_tokens = {}

    for chunks in chunk_itr:
        # Run pre-tokenization on the chunks and store the chunk counts for each pre-token
        chunk_pre_tokens_list = pool.starmap(get_chunk_pre_token_counts, zip(chunks, repeat(special_tokens)))
        
        # Merge the chunk pre-tokens and counts into one dict
        for chunk_pre_tokens in chunk_pre_tokens_list:
            for pre_token, count in chunk_pre_tokens.items():
                if pre_token in pre_tokens:
                    pre_tokens[pre_token] += count
                else:
                    pre_tokens[pre_token] = count
        
    return pre_tokens