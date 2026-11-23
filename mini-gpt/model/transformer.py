import torch
import torch.nn as nn
from model.attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.debug = True

    def forward(self, x):
        norm1 = self.ln1(x)
        attn_out = self.attn(norm1)
        x = x + attn_out  # Residual 1

        norm2 = self.ln2(x)
        expanded = self.ffn[0](norm2)      # Linear(16 -> 64)
        activated = self.ffn[1](expanded)  # ReLU
        compressed = self.ffn[2](activated)# Linear(64 -> 16)
        x = x + compressed                 # Residual 2

        if self.debug:
            print(f"\n  [5. TRANSFORMER BLOCK] LayerNorm1 -> Self-Attention -> Residual Add 1 complete.")
            print(f"      FFN Expand: {norm2.shape} -> {expanded.shape} -> ReLU zeroed {(activated == 0).sum().item()}/{activated.numel()} values -> Compress: {compressed.shape}")
            print(f"      Residual Add 2 complete. Block output shape: {x.shape}")
            self.debug = False
        return x
