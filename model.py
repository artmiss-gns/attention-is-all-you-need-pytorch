import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

class Attention(nn.Module):
    def __init__(self, model_dim, dropout=None):
        super().__init__()

        self.w_key = nn.Linear(model_dim, model_dim, bias=False)
        self.w_query = nn.Linear(model_dim, model_dim, bias=False)
        self.w_value = nn.Linear(model_dim, model_dim, bias=False)
        self.scale = model_dim ** 0.5
        self.dropout = dropout

    def forward(self, batch, encoder_output=None, mask=None):
        '''
        if encoder output is None, it calculates self-attention; otherwise, cross-attention
        '''
        key = self.w_key(encoder_output) if (encoder_output is not None) else self.w_key(batch)
        query = self.w_query(batch)
        value = self.w_value(encoder_output) if (encoder_output is not None) else self.w_value(batch)

        attention_scores = query @ key.transpose(-2, -1)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.bool(), float('-inf'))
        # [batch_size, seq_length, seq_length]

        attention_scores = F.softmax(attention_scores / self.scale, dim=-1)

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        return attention_scores @ value

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.attention = Attention(model_dim, self.dropout)
        self.layer_norm_attn = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm_ffn = nn.LayerNorm(normalized_shape=model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim*4),
            nn.ReLU(),
            nn.Linear(model_dim*4, model_dim),
        )

    def add_and_norm(self, input_embedding, attention_vector, norm):
        return norm(
            input_embedding + self.dropout(attention_vector)
        )

    def forward(self, x, mask=None):
        attention_vector = self.attention(x, mask=mask)
        x = self.add_and_norm(x, attention_vector, norm=self.layer_norm_attn)
        ffn_output = self.ffn(x)
        x = self.add_and_norm(x, ffn_output, norm=self.layer_norm_ffn)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.self_attention = Attention(model_dim, self.dropout)
        self.cross_attention = Attention(model_dim, self.dropout)
        self.layer_norm_self_attn = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm_cross_attn = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm_fnn = nn.LayerNorm(normalized_shape=model_dim)
        self.fnn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )

    def add_and_norm(self, input_embedding, attention_vector, norm):
        return norm(
            input_embedding + self.dropout(attention_vector)
        )

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # calculating the self-attention
        seq_len = x.shape[-2]
        triangle_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device=x.device)
        self_attention_vector = self.self_attention(x, mask=(triangle_mask | tgt_mask.to(x.device)))
        x = self.add_and_norm(x, self_attention_vector, norm=self.layer_norm_self_attn)

        # calculating the cross-attention
        seq_len = x.shape[1]
        cross_attention_vector = self.cross_attention(x, encoder_output=encoder_output, mask=src_mask)
        x = self.add_and_norm(x, cross_attention_vector, norm=self.layer_norm_cross_attn)

        # FNN layer
        ffn_output = self.fnn(x)
        x = self.add_and_norm(x, ffn_output, norm=self.layer_norm_fnn)

        return x

class Transformer(nn.Module):
    def __init__(self, model_dim, num_layers, src_vocab_size, tgt_vocab_size, padding_idx, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.padding_idx = padding_idx

        self.encoder_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=model_dim,
            padding_idx=padding_idx
        )

        self.decoder_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=model_dim,
            padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(p=dropout)

        # NOTE: In the original paper, they used one embedding for both the encoder and decoder, to reduce the model's weights and more optimization
        # in this implementation, we used 2 different embeddings for simplicity

        self.positional_encoder = PositionalEncoder(model_dim, max_len=5000)

        self.encoder_layers = nn.ModuleList(EncoderLayer(model_dim, self.dropout) for _ in range(num_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(model_dim, self.dropout) for _ in range(num_layers))

        self.fc_out = nn.Linear(model_dim, tgt_vocab_size)
        # NOTE: In the original paper, they used the embedding for the output too, since they realized these are just transpose of each other
        # Embedding: Maps "Word ID" -> "Vector"
        # Final Layer: Maps "Vector" -> "Word Probability"

    def forward(self, src, tgt):
        # --- Embedding --- #
        src_embedding = self.encoder_embedding(src) * math.sqrt(self.model_dim)
        tgt_embedding = self.decoder_embedding(tgt) * math.sqrt(self.model_dim)
        # [batch_size, seq_len, model_dim]

        # --- Positional Encoding --- #
        src_embedding = self.positional_encoder(src_embedding)
        tgt_embedding = self.positional_encoder(tgt_embedding)
        # [batch_size, seq_len, model_dim]

        # Dropout
        src_embedding = self.dropout(src_embedding)
        tgt_embedding = self.dropout(tgt_embedding)

        # --- Encoder ---
        src_mask = (src == self.padding_idx) # padding mask
        src_mask = src_mask.unsqueeze(1) # ! for multi-head attention, we should unsqueeze(2) in addition to what we have
        # [batch_size, 1, seq_len]
        for encoder in self.encoder_layers:
            src_embedding = encoder(src_embedding, mask=src_mask)
        encoder_outputs = src_embedding # just a rename for better readability

        # --- Decoder ---
        tgt_mask = (tgt == self.padding_idx) # padding mask
        tgt_mask = tgt_mask.unsqueeze(1)
        for decoder in self.decoder_layers:
            tgt_embedding = decoder(tgt_embedding, encoder_outputs, tgt_mask=tgt_mask, src_mask=src_mask)
        decoder_outputs = tgt_embedding

        return self.fc_out(decoder_outputs)
