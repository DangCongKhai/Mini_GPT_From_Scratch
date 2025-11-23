import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, context_size):
        super().__init__()

        i = (torch.arange(d_model) // 2).view(1, -1)
        positions = torch.arange(context_size).view(-1, 1)

        encoding = positions / (10000 ** (2 * i / d_model))
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        encoding = encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", encoding)

    def forward(self, X):
        # X with shape: B x N x d_model
        N = X.shape[1]
        return X + self.positional_encoding[:, :N, :]


class Self_Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, context_size):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

        masked_matrix = torch.ones(context_size, context_size)
        masked_matrix = torch.tril(masked_matrix, diagonal=0)
        self.register_buffer("masked_matrix", masked_matrix)

    def forward(self, X, attention_mask=None):
        # X has shape: B x N x D_model
        N = X.shape[1]
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)  # B X N x D_V
        Z = Q @ K.permute(0, 2, 1) / math.sqrt(self.d_k)

        if attention_mask is not None:
            attention_mask = attention_mask & self.masked_matrix[:N, :N]
        else:
            attention_mask = (self.masked_matrix[:N, :N]).bool()

        Z = Z.masked_fill(~attention_mask, float("-inf"))
        return torch.matmul(torch.softmax(Z, dim=-1), V)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_size, dropout_rate):
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must be divided by d_model"

        d_head = d_model // num_heads
        self.num_heads = num_heads
        self.attentions = nn.ModuleList(
            [
                Self_Attention(d_model, d_head, d_head, context_size)
                for i in range(self.num_heads)
            ]
        )
        self.output = nn.Linear(num_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X, attention_mask=None):
        outputs = []
        for attention in self.attentions:
            output = attention(X, attention_mask)
            outputs.append(output)
        X = torch.cat(outputs, dim=-1)
        X = self.output(X)
        X = self.dropout(X)
        return X


class FFN(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super().__init__()
        hidden_dim = input_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        X = nn.GELU()(self.fc1(X))
        X = self.fc2(X)
        X = self.dropout(X)
        return X


class GPT_Block(nn.Module):
    def __init__(self, d_model, context_size, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.masked_attention = MaskedMultiHeadAttention(
            d_model, num_heads, context_size, dropout_rate
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout_rate)

    def forward(self, X, attention_mask=None):
        residual = X
        X = self.masked_attention(self.layer_norm1(X), attention_mask) + residual

        residual = X
        X = self.ffn(self.layer_norm2(X)) + residual
        return X


class MiniGPT(nn.Module):
    def __init__(self, Nx, d_model, num_heads, vocab_size, dropout_rate, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, context_size)
        self.gpt_blocks = nn.ModuleList(
            [
                GPT_Block(d_model, context_size, num_heads, dropout_rate)
                for _ in range(Nx)
            ]
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, X, attention_mask=None):
        X = self.positional_embedding(self.embedding(X))
        for gpt_block in self.gpt_blocks:
            X = gpt_block(X, attention_mask)
        X = self.norm_final(X)
        X = self.output(X)
        return X
