from cs336_basics.tokenizer import Tokenizer
import pickle
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Tokenization',
                    description='Tokenize an input file and store the vocab and merges to disk')
    parser.add_argument("--vocab", required=True, type=str)
    parser.add_argument("--merges", required=True, type=str)
    parser.add_argument("--input", default=None, type=str)
    args = parser.parse_args()

    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    with open(args.input, errors="replace") as f:
        encoded_ids = []

        start_time = time.time()

        for id in  tokenizer.encode_iterable(f):
            encoded_ids.append(id)

        end_time = time.time()
        duration_sec = end_time - start_time

    with open(args.input, errors="replace") as f:
        content_encoded = f.read().encode("utf-8")

    compression_ratio = len(content_encoded) / len(encoded_ids)
    throughput_mb = len(content_encoded) / (duration_sec * 1024**2)

    print(f"Compression ratio (bytes/tokens): {compression_ratio}")
    print(f"Throughput (MB/sec): {throughput_mb}")