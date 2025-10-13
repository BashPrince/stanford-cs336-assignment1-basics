import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, type=str)
args = parser.parse_args()

with open(args.vocab, 'rb') as file:
    vocab = pickle.load(file)

tokens = vocab.values()
byte_sorted_tokens = sorted(tokens, reverse=True, key=lambda t: len(t))

n = 50
print(f"Longest {n} tokens by byte length:\n")

for t in byte_sorted_tokens[:n]:
    try:
        print(t.decode('utf-8'))
    except UnicodeDecodeError:
        print("!!!UnicodeDecodeError!!!")