import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, context_size):
        super().__init__()

        i = (torch.arange(d_model) // 2).view(1, -1)
        positions = torch.arange(context_size).view(-1, 1)

        self.encoding = positions / (10000 ** (2 * i / d_model))
        self.encoding[:, 0::2] = torch.sin(self.encoding[:, 0::2])
        self.encoding[:, 1::2] = torch.cos(self.encoding[:, 1::2])
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, X):
        # X with shape: B x N x d_model
        N = X.shape[1]
        print(self.encoding.shape)
        return X + self.encoding[:, :N, :]


class Self_Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, masked=False):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.masked = masked

    def forward(self, X):
        # X has shape: B x N x D_model
        N = X.shape[1]
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)  # B X N x D_V

        Z = Q @ K.permute(0, 2, 1) / math.sqrt(self.d_k)

        masked_matrix = torch.ones(N, N)
        masked_matrix = torch.triu(masked_matrix, diagonal=1) * (-1e8)

        if self.masked:
            Z = Z + masked_matrix
        return torch.matmul(torch.softmax(Z, dim=-1), V)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attentions = [
            Self_Attention(d_model, d_k, d_v, masked=True)
            for i in range(self.num_heads)
        ]
        self.output = nn.Linear(num_heads * d_v, d_model)

    def forward(self, X):
        X = torch.cat([attention(X) for attention in self.attentions], dim=-1)
        return self.output(X)


class FFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, X):
        X = nn.GELU()(self.fc1(X))
        return self.fc2(X)


class GPT_Block(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.masked_attention = MaskedMultiHeadAttention(d_model, d_k, d_v, num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        X = self.dropout(self.masked_attention(self.layer_norm1(X))) + X
        X = self.dropout(self.ffn(self.layer_norm2(X))) + X
        return X


class MiniGPT(nn.Module):
    def __init__(self, Nx, d_model, num_heads, vocab_size, dropout_rate, context_size):
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must be divided by d_model"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, context_size)
        d_k = d_v = d_model // num_heads
        self.gpt_blocks = nn.Sequential(
            *[GPT_Block(d_model, d_k, d_v, num_heads, dropout_rate) for _ in range(Nx)]
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        X = self.positional_embedding(self.embedding(X))
        X = self.gpt_blocks(X)
        X = self.output(X)
        return X
