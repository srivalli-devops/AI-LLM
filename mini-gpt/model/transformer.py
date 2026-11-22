import torch.nn as nn
from model.attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        # Expand (4x) then compress back down, just like we discussed!
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

