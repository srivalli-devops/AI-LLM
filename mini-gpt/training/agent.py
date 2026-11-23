import subprocess
from datetime import datetime
import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

# 1. Define Real VM Tools (The Agent's "Hands")
def cat_tool():
    timestamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[{timestamp}] CAT DETECTED -> Logged sleep event to vm_events.log\n"
    with open("vm_events.log", "a") as f:
        f.write(msg)
    print(f"\n  >>> [AGENT TOOL TRIGGERED: cat_tool()] {msg.strip()}")

def dog_tool():
    uptime = subprocess.check_output(["uptime", "-p"]).decode().strip()
    msg = f"DOG DETECTED -> Checked VM Uptime: {uptime}"
    print(f"\n  >>> [AGENT TOOL TRIGGERED: dog_tool()] {msg}")

# 2. Load your EXISTING Cat/Dog MiniGPT (The Agent's "Brain")
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
model.load_state_dict(torch.load("mini_gpt.pth", weights_only=True))
model.eval()

# 3. Start generating from "the " and watch for trigger words!
prompt = "the "
idx = torch.tensor([tokenizer.encode(prompt)])
generated_text = prompt
triggered_events = set()  # Prevent firing the same tool twice for the same word

print(f"[USER PROMPT]: '{prompt}'")
print("[AGENT STREAMING]: ", end="", flush=True)

for _ in range(48):
    logits = model(idx)
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)

    char = tokenizer.id_to_char[next_id.item()]
    generated_text += char
    print(char, end="", flush=True)

    # 4. Agent Watcher: Intercept keywords in the stream and execute VM tools!
    if "cat" in generated_text and "cat_done" not in triggered_events:
        cat_tool()
        triggered_events.add("cat_done")
        print("[AGENT RESUMING]: ", end="", flush=True)

    if "dog" in generated_text and "dog_done" not in triggered_events:
        dog_tool()
        triggered_events.add("dog_done")
        print("[AGENT RESUMING]: ", end="", flush=True)

print(f"\n\n[FINAL STORY]: {generated_text}")
