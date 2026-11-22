import torch.nn as nn
from model.embeddings import Embeddings
from model.transformer import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embed = Embeddings(vocab_size, embed_dim)
        self.block = TransformerBlock(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        x = self.block(x)
        logits = self.lm_head(x)  # Scores for every character in vocab
        return logits
