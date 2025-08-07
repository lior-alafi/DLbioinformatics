import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_layers=2, num_heads=4,
                 hidden_dim=256, dropout=0.1, output_dim=1):
        super(TransformerEncoderModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, rna_tensor, rbp_tensor, rna_mask, rbp_mask):
        # Shape: (B, L)
        rna_embed = self.embedding(rna_tensor)       # (B, L, D)
        rbp_embed = self.embedding(rbp_tensor)       # (B, L, D)

        # Create attention masks for Transformer (True = ignore, False = attend)
        rna_attn_mask = ~rna_mask.bool()  # invert 1↔0
        rbp_attn_mask = ~rbp_mask.bool()

        rna_encoded = self.transformer_encoder(rna_embed, src_key_padding_mask=rna_attn_mask)  # (B, L, D)
        rbp_encoded = self.transformer_encoder(rbp_embed, src_key_padding_mask=rbp_attn_mask)

        # Pool: Mean over sequence dimension
        rna_pooled = rna_encoded.mean(dim=1)   # (B, D)
        rbp_pooled = rbp_encoded.mean(dim=1)   # (B, D)

        combined = torch.cat([rna_pooled, rbp_pooled], dim=1)  # (B, 2D)
        output = self.fc(combined)  # (B, 1)

        return output.squeeze(1)  # (B,)
