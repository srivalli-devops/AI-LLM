import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, block_size, embedding_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

    def forward(self, positions):

        return self.embedding(positions)


if __name__ == "__main__":

    block_size = 8
    embedding_dim = 8

    positional_embedding = PositionalEmbedding(
        block_size,
        embedding_dim
    )

    # Four character positions: 0, 1, 2, 3
    positions = torch.arange(4)

    position_vectors = positional_embedding(positions)

    print("Positions:", positions)

    print("Position embedding shape:",
          position_vectors.shape)

    print("Position vectors:")
    print(position_vectors)
