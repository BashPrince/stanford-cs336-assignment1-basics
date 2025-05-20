import pickle
import argparse
from pathlib import Path
from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Tokenizer Training',
                    description='Tokenize an input file and store the vocab and merges to disk')
    parser.add_argument("--corpus", required=True, type=str)
    args = parser.parse_args()
    
    vocab, merge_sequence = train_bpe(input_path=args.corpus, vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes=None)

    file_stem = Path(args.corpus).stem

    with open(f"{file_stem}-vocab.pickle", 'ab') as file:
        pickle.dump(vocab, file=file)
        
    with open(f"{file_stem}-merges.pickle", 'ab') as file:
        pickle.dump(merge_sequence, file=file)