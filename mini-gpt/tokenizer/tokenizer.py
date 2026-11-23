class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        print(f"\n[1. TOKENIZER INIT] Found {self.vocab_size} unique characters.")
        print(f"    Full char_to_id dictionary: {self.char_to_id}")

    def encode(self, text):
        ids = [self.char_to_id[c] for c in text]
        print(f"[2. TOKENIZER ENCODE] Input text: '{text}' (len={len(text)})")
        print(f"    Encoded IDs: {ids}")
        return ids

    def decode(self, ids):
        text = "".join([self.id_to_char[i] for i in ids])
        print(f"\n[10. TOKENIZER DECODE] Decoding {len(ids)} IDs: {ids[:8]}... -> '{text}'")
        return text
