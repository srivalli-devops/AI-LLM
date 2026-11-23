import numpy as np
import onnxruntime as ort
from tokenizer.tokenizer import SimpleTokenizer

# 1. Load tokenizer & start the ONNX Runtime CPU engine (No PyTorch needed!)
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

session = ort.InferenceSession("mini_gpt.onnx", providers=["CPUExecutionProvider"])

# 2. Start with your prompt as a NumPy int64 array
prompt = "the cat"
idx = np.array([tokenizer.encode(prompt)], dtype=np.int64)

# 3. Generate 45 characters using ONNX Runtime
for _ in range(45):
    logits = session.run(["logits"], {"input_ids": idx})[0]  # Extract array from list
    last_char_logits = logits[:, -1, :]
    next_id = np.argmax(last_char_logits, axis=-1, keepdims=True)
    idx = np.concatenate([idx, next_id], axis=1)

print("\nONNX Runtime Output:\n", tokenizer.decode(idx[0].tolist()))
