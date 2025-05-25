import pickle
import argparse
from pathlib import Path
from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Tokenizer Training',
                    description='Train a tokenizer on an input file and store the vocab and merges to disk')
    parser.add_argument("--corpus", required=True, type=str)
    parser.add_argument("--vocab_size", required=True, type=int)
    parser.add_argument("--num_processes", default=None, type=int)
    args = parser.parse_args()
    
    vocab, merge_sequence = train_bpe(
        input_path=args.corpus,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        num_processes=args.num_processes)

    file_stem = Path(args.corpus).stem

    with open(f"{file_stem}-vocab.pickle", 'ab') as file:
        pickle.dump(vocab, file=file)
        
    with open(f"{file_stem}-merges.pickle", 'ab') as file:
        pickle.dump(merge_sequence, file=file)