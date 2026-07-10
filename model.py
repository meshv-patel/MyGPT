from pyexpat import features

import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, X):
        return self.embedding(X) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, X):
        X = X + self.pe[:, :X.size(1), :]
        return self.dropout(X)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        return self.alpha * (X - mean) / (std + self.epsilon) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, X):
        return self.network(X)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout =  None):
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e09)

        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores 

    def forward(self, Q, K, V, mask=None):
        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)

        X, attention_scores = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        X = X.transpose(1, 2).contiguous().view(X.size(0), -1, self.d_model)

        return self.w_o(X), attention_scores
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization(features)

    def forward(self, X, sublayer):
        return self.normalization(X + self.dropout(sublayer(self.normalization(X))))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, attention: MultiHeadAttention, ffn: FeedForward, dropout: float) -> None:
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, X, mask):
        X = self.residual_connections[0](X, lambda X: self.attention(X, X, X, mask))
        X = self.residual_connections[1](X, self.ffn)
        return X

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, X, mask):
        for layer in self.layers:
            X = layer(X, mask)
        return self.norm(X)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, ffn: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffn = ffn
        self.residual_connections = nn.ModuleList([ResidualConnection(self_attention.d_model, dropout) for _ in range(3)])

    def forward(self, X, encoder_output, src_mask, tgt_mask):
        X = self.residual_connections[0](X, lambda X: self.self_attention(X, X, X, tgt_mask))
        X = self.residual_connections[1](X, lambda X: self.cross_attention(X, encoder_output, encoder_output, src_mask))
        X = self.residual_connections[2](X, self.ffn)
        return X

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, X, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            X = layer(X, encoder_output, src_mask, tgt_mask)
        return self.norm(X)
        
class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        return self.log_softmax(self.project(X), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,projection: Projection) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def projection(self, X):
        return self.projection(X)

def make_model(src_vocab: int, tgt_vocab: int, src_len: int, tgt_len: int, d_model: int = 512, d_ff: int = 2048, h: int = 8, num_layers: int = 6, dropout: float = 0.1) -> Transformer:
    src_embedding = InputEmbedding(d_model, src_vocab)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab)

    src_pos = PositionalEncoding(d_model, src_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_len, dropout)

    encoder_blocks = []
    for _ in range(num_layers):
        attention = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(d_model, attention, ffn, dropout))
    
    decoder_blocks = []
    for _ in range(num_layers):
        self_attention = MultiHeadAttention(d_model, h, dropout)
        cross_attention = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attention, cross_attention, ffn, dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = Projection(d_model, tgt_vocab)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer