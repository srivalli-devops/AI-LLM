import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, idx):
        seq_len = idx.shape[1]
        positions = torch.arange(seq_len)
        # Combine word meaning + word position
        return self.token_emb(idx) + self.pos_emb(positions)
