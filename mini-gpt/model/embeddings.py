
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):

        return self.embedding(token_ids)


if __name__ == "__main__":

    vocab_size = 33
    embedding_dim = 8

    # Create embedding layer
    embedding_layer = TokenEmbedding(
        vocab_size,
        embedding_dim
    )

    # Token IDs from our tokenizer
    token_ids = torch.tensor([3, 25, 27, 18])

    # Convert token IDs into vectors
    embeddings = embedding_layer(token_ids)

    print("Token IDs:", token_ids)

    print("Embedding shape:", embeddings.shape)

    print("Embedding vectors:")
    print(embeddings)
