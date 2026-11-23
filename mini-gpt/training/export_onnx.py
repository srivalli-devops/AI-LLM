import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

# 1. Load your trained PyTorch model
text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
model.load_state_dict(torch.load("mini_gpt.pth", weights_only=True))
model.eval()

# 2. Create a dummy input tensor so PyTorch can trace the math graph
dummy_input = torch.tensor([tokenizer.encode("the cat")], dtype=torch.long)

# 3. Export to mini_gpt.onnx (allowing dynamic sequence lengths as words grow)
torch.onnx.export(
    model,
    dummy_input,
    "mini_gpt.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "seq_len"},
        "logits": {1: "seq_len"}
    },
    opset_version=14
)
print("Successfully exported mini_gpt.onnx!")
