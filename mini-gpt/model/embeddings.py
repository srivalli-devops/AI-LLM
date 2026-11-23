import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.debug = True

    def forward(self, idx):
        seq_len = idx.shape[1]
        positions = torch.arange(seq_len)
        tok = self.token_emb(idx)
        pos = self.pos_emb(positions)
        out = tok + pos
        if self.debug:
            print(f"\n  [3. EMBEDDINGS] Input IDs shape: {idx.shape} | Positions tensor: {positions.tolist()[:8]}...")
            print(f"      Token 0 (ID={idx[0,0].item()}) Token Emb (first 4 dims): {torch.round(tok[0, 0, :4], decimals=3).tolist()}")
            print(f"      Token 0 (Pos=0) Pos Emb (first 4 dims):   {torch.round(pos[0, :4], decimals=3).tolist()}")
            print(f"      Combined Output (tok + pos) shape: {out.shape} | First 4 dims: {torch.round(out[0, 0, :4], decimals=3).tolist()}")
            self.debug = False
        return out
