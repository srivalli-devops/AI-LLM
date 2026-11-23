import torch
import torch.nn.functional as F
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

print("==========================================")
print("       MINI GPT TRAINING TRACE")
print("==========================================")
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n[2.2 MODEL CREATED] Total learnable parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

inputs = data[:, :-1]
targets = data[:, 1:]
print(f"[2.5 SHIFTED DATA] Inputs shape: {inputs.shape} ('{tokenizer.decode(inputs[0,:6].tolist())}...')")
print(f"                   Targets shape: {targets.shape} ('{tokenizer.decode(targets[0,:6].tolist())}...')")

for step in range(300):
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()

    if step == 0:
        w_before = model.lm_head.weight[0, :4].clone().detach()
        grad_sample = model.lm_head.weight.grad[0, :4].clone().detach()
        print(f"\n[7. BACKPROP & WEIGHT UPDATE (Step 1)]")
        print(f"    Initial Cross-Entropy Loss: {loss.item():.4f}")
        print(f"    lm_head.weight[0,:4] BEFORE step: {torch.round(w_before, decimals=4).tolist()}")
        print(f"    lm_head.weight.grad[0,:4]:        {torch.round(grad_sample, decimals=4).tolist()}")

    optimizer.step()

    if step == 0:
        w_after = model.lm_head.weight[0, :4].clone().detach()
        print(f"    lm_head.weight[0,:4] AFTER step:  {torch.round(w_after, decimals=4).tolist()}")
        print("\n--- Running remaining 299 training steps ---")

    if (step + 1) % 100 == 0:
        print(f"Step {step+1:3d} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mini_gpt.pth")
print("=== SAVED TRAINED WEIGHTS TO mini_gpt.pth ===\n")
