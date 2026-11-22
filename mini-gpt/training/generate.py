import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

# 1. Load tokenizer & trained model
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
model.load_state_dict(torch.load("mini_gpt.pth", weights_only=True))
model.eval()

# 2. Start with a prompt and generate 35 new characters
prompt = "the cat"
idx = torch.tensor([tokenizer.encode(prompt)])

for _ in range(35):
    logits = model(idx)
    last_char_logits = logits[:, -1, :]               # Focus on the newest character
    next_id = torch.argmax(last_char_logits, dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)            # Append predicted char
