import torch
import torch.nn.functional as F
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

# 1. Load data & initialize tokenizer + model
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Shape: (1, seq_len)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# 2. Train for 300 steps (input = chars 0..N-1, target = chars 1..N)
inputs = data[:, :-1]
targets = data[:, 1:]

for step in range(300):
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 50 == 0:
        print(f"Step {step+1:3d} | Loss: {loss.item():.4f}")

# 3. Save trained weights
torch.save(model.state_dict(), "mini_gpt.pth")
print("Saved trained model to mini_gpt.pth!")
