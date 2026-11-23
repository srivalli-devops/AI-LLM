import torch
import torch.nn.functional as F
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

print("==========================================")
print("      MINI GPT GENERATION TRACE")
print("==========================================")
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
model.load_state_dict(torch.load("mini_gpt.pth", weights_only=True))
model.eval()

prompt = "the cat"
idx = torch.tensor([tokenizer.encode(prompt)])

print("\n[8. AUTOREGRESSIVE GENERATION LOOP (Showing first 3 steps in detail)]")
for i in range(45):
    logits = model(idx)
    last_char_logits = logits[:, -1, :]
    probs = F.softmax(last_char_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=3)
    next_id = torch.argmax(last_char_logits, dim=-1, keepdim=True)

    if i < 3:
        top_choices = [(tokenizer.id_to_char[tid.item()], round(tp.item()*100, 1)) for tid, tp in zip(top_ids[0], top_probs[0])]
        print(f"    Gen Step {i+1} | Context: '{tokenizer.decode(idx[0].tolist())}'")
        print(f"               | Top 3 Predictions (%): {top_choices} -> Picked: '{tokenizer.id_to_char[next_id.item()]}'")

    idx = torch.cat([idx, next_id], dim=1)

print("\nFinal Generated Output:\n", tokenizer.decode(idx[0].tolist()))
