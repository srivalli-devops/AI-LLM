import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.debug = True

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.query(x), self.key(x), self.value(x)
        raw_scores = (Q @ K.transpose(-2, -1)) / (C ** 0.5)
        
        mask = torch.tril(torch.ones(T, T))
        masked_scores = raw_scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(masked_scores, dim=-1)
        out = weights @ V

        if self.debug:
            print(f"\n    [4. SELF-ATTENTION] Input x shape: {x.shape}")
            print(f"        Q[0,0,:4]: {torch.round(Q[0,0,:4], decimals=3).tolist()} | K[0,0,:4]: {torch.round(K[0,0,:4], decimals=3).tolist()}")
            print(f"        Raw Scores (before mask, top-left 3x3):\n{torch.round(raw_scores[0, :3, :3], decimals=2)}")
            print(f"        Masked Scores (future = -inf, top-left 3x3):\n{torch.round(masked_scores[0, :3, :3], decimals=2)}")
            print(f"        Softmax Attention Percentages (rows sum to 1.0, top-left 3x3):\n{torch.round(weights[0, :3, :3], decimals=2)}")
            print(f"        Context Output (weights @ V) shape: {out.shape}")
            self.debug = False
        return out
