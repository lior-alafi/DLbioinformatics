import math
import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):  # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1, L, D)

def masked_mean(x, mask):  # x: (B, L, D); mask: (B, L) with 1=real, 0=pad
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
    x = x * mask.unsqueeze(-1)  # zero out pads
    return x.sum(dim=1) / denom  # (B, D)

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_layers=2, num_heads=4,
                 hidden_dim=256, dropout=0.1, output_dim=1, twin_encoders=True, max_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.posenc = SinusoidalPositionalEncoding(embedding_dim, max_len=max_len)

        def make_enc():
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        # separate encoders by default (safer for RNA vs RBP)
        self.rna_encoder = make_enc()
        self.rbp_encoder = make_enc() if twin_encoders else self.rna_encoder

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 768),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, rna_tensor, rbp_tensor, rna_mask, rbp_mask):
        # embeds + PE
        rna = self.posenc(self.embedding(rna_tensor))
        rbp = self.posenc(self.embedding(rbp_tensor))

        # key_padding_mask expects True for PAD positions
        rna_kpm = ~rna_mask.bool()
        rbp_kpm = ~rbp_mask.bool()

        rna_enc = self.rna_encoder(rna, src_key_padding_mask=rna_kpm)
        rbp_enc = self.rbp_encoder(rbp, src_key_padding_mask=rbp_kpm)

        rna_pooled = masked_mean(rna_enc, rna_mask)  # (B, D)
        rbp_pooled = masked_mean(rbp_enc, rbp_mask)  # (B, D)

        out = self.fc(torch.cat([rna_pooled, rbp_pooled], dim=1)).squeeze(1)  # (B,)
        return out
