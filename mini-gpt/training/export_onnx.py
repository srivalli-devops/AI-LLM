import torch
from tokenizer.tokenizer import SimpleTokenizer
from model.gpt import MiniGPT

text = open("dataset/training_data.txt").read()
tokenizer = SimpleTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size, embed_dim=16)
model.load_state_dict(torch.load("mini_gpt.pth", weights_only=True))
model.eval()

# Turn off debug prints so they aren't baked into the trace
model.embed.debug = False
model.block.debug = False
model.block.attn.debug = False
model.debug = False

dummy_input = torch.tensor([tokenizer.encode("the cat")], dtype=torch.long)

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
    opset_version=14,
    dynamo=False
)
print("Successfully exported mini_gpt.onnx!")
