from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes=1)