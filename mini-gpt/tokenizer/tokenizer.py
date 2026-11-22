
class CharTokenizer:

    def __init__(self, text):

        # Extract unique characters
        self.chars = sorted(set(text))

        # Character to integer mapping
        self.stoi = {
            ch: i for i, ch in enumerate(self.chars)
        }

        # Integer to character mapping
        self.itos = {
            i: ch for i, ch in enumerate(self.chars)
        }

        # Total vocabulary size
        self.vocab_size = len(self.chars)

    def encode(self, text):

        # Convert characters into integers
        return [
            self.stoi[ch] for ch in text
        ]

    def decode(self, tokens):

        # Convert integers back into characters
        return "".join(
            self.itos[token] for token in tokens
        )


if __name__ == "__main__":

    with open("dataset/training_data.txt", "r",
              encoding="utf-8") as file:

        text = file.read()

    tokenizer = CharTokenizer(text)

    sample = "Artificial intelligence"

    encoded = tokenizer.encode(sample)

    decoded = tokenizer.decode(encoded)

    print("Original text:", sample)

    print("Vocabulary:", tokenizer.chars)

    print("Vocabulary size:", tokenizer.vocab_size)

    print("Encoded tokens:", encoded)

    print("Decoded text:", decoded)
