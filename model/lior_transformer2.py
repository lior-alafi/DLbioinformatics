import torch
from torch import nn

from model.lior_transformer import masked_mean, SinusoidalPositionalEncoding


class TransformerEncoderModelV2(nn.Module):
    # ***
    # *** עיקר השינוי: ארכיטקטורת אינטראקציה מוקדמת ***
    # *** הרצפים מאוחדים לפני הכניסה ל-Transformer
    # ***
    def __init__(self, vocab_size, embedding_dim=128, num_layers=2, num_heads=4,
                 hidden_dim=256, dropout=0.1, output_dim=1, max_len=4096): # twin_encoders הוסר
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.posenc = SinusoidalPositionalEncoding(embedding_dim, max_len=max_len)

        def make_enc():
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        # Encoder יחיד שמטפל ברצף המאוחד
        self.encoder = make_enc()

        # שכבת FC מעודכנת לקבל פלט מאוחד
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, output_dim)
        )

    def forward(self, rna_tensor, rbp_tensor, rna_mask, rbp_mask):
        # 1. הטמעה
        rna_embed = self.embedding(rna_tensor)
        rbp_embed = self.embedding(rbp_tensor)

        # 2. איחוד הרצפים והמסיכות
        combined_embeds = torch.cat([rna_embed, rbp_embed], dim=1)
        combined_mask = torch.cat([rna_mask, rbp_mask], dim=1)

        # 3. הוספת קידוד מיקום
        x = self.posenc(combined_embeds)

        # 4. הכנת מסיכת ריפוד
        combined_kpm = ~combined_mask.bool()

        # 5. העברה ב-Encoder המאוחד
        encoded = self.encoder(x, src_key_padding_mask=combined_kpm)

        # 6. Pooling
        pooled = masked_mean(encoded, combined_mask)

        # 7. שכבה אחרונה
        out = self.fc(pooled).squeeze(1)
        return out