import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.query(x), self.key(x), self.value(x)

        scores = (Q @ K.transpose(-2, -1)) / (C ** 0.5)
        # Causal mask: prevent peeking at future characters
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ V
