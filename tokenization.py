from cs336_basics.tokenizer import Tokenizer
import os
import sys
import pickle
import argparse
import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Tokenization',
                    description='Tokenize an input file and store the vocab and merges to disk')
    parser.add_argument("--vocab", required=True, type=str)
    parser.add_argument("--merges", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    temp_arrays = []

    with open(args.input, errors="replace") as f:
        # Get total file size in bytes
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        encoded_ids = []
        total_num_ids = 0

        start_time = time.time()

        for id in  tokenizer.encode_iterable(f, file_size):
            encoded_ids.append(id)

            # Periodically convert ids to np.uint16 to save memory
            if sys.getsizeof(encoded_ids) > 1024**3:
                encoded_ids_np = np.array(encoded_ids, dtype=np.uint16)
                temp_arrays.append(encoded_ids_np)
                total_num_ids += len(encoded_ids)
                encoded_ids = []

        end_time = time.time()

        duration_sec = end_time - start_time

    total_num_ids += len(encoded_ids)
    compression_ratio = file_size / total_num_ids
    throughput_mb = file_size / (duration_sec * 1024**2)

    print(f"Compression ratio (bytes/tokens): {compression_ratio}")
    print(f"Throughput (MB/sec): {throughput_mb}")

    if args.output:
        if encoded_ids:
            temp_arrays.append(np.array(encoded_ids, dtype=np.uint16))

        cat_array = np.concatenate(temp_arrays)
        np.save(args.output, cat_array)