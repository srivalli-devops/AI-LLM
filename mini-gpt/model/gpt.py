import torch
import torch.nn as nn
from model.embeddings import Embeddings
from model.transformer import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embed = Embeddings(vocab_size, embed_dim)
        self.block = TransformerBlock(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.debug = True

    def forward(self, idx):
        x = self.embed(idx)
        x = self.block(x)
        logits = self.lm_head(x)
        if self.debug:
            print(f"\n  [6. LM_HEAD PROJECTION] Final Linear(16 -> {logits.shape[-1]}) produced logits shape: {logits.shape}")
            print(f"      Token 0 raw logits (first 6 vocab scores): {torch.round(logits[0, 0, :6], decimals=2).tolist()}")
            self.debug = False
        return logits
