class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id[c] for c in text]

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids])
