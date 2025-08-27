import torch
import torch.nn as nn
import torch.nn.functional as F


class RecommendationModelV2(nn.Module):
    def __init__(
            self,
            nucleotide_vocab,
            amino_vocab,
            hidden_size,
            embedding_dim,
            fc_hidden_size,
            lstm_layers,
            bidirectional=False,
    ):
        super(RecommendationModelV2, self).__init__()
        self.AGCU_V = nucleotide_vocab
        self.AA_V = amino_vocab
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # Embedding layers
        self.AGCU_embedding = nn.Embedding(len(self.AGCU_V), self.embedding_dim, padding_idx=0)
        self.AA_embedding = nn.Embedding(len(self.AA_V), self.embedding_dim, padding_idx=0)

        # LSTM layers
        self.AGCU_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=self.bidirectional
        )

        self.AA_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=self.bidirectional,
        )

        # Because we use concat(rna, rbp, mul, diff)
        self.merge_dim = hidden_size * 4
        if self.bidirectional:
            self.merge_dim = self.merge_dim * 2

        # Fully connected layers
        self.fc1 = nn.Linear(self.merge_dim, fc_hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size * 2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(fc_hidden_size * 2, 1)  # output size 1 (binding strength)

    def forward(self, nec_seq, rpb_seq, nec_mask, rpb_mask):
        # Embedding
        rpb_embedded = self.AA_embedding(rpb_seq)
        nec_embedded = self.AGCU_embedding(nec_seq)

        # Bi-LSTM
        rpb_out, _ = self.AA_lstm(rpb_embedded)  # [B, T, H]
        nec_out, _ = self.AGCU_lstm(nec_embedded)  # [B, T, H]

        # Mean with mask
        rpb_repr = self.masked_mean(rpb_out, rpb_mask)
        nec_repr = self.masked_mean(nec_out, nec_mask)

        # Enrich representation
        interaction_mul = nec_repr * rpb_repr
        interaction_diff = torch.abs(nec_repr - rpb_repr)

        # Merge features
        combined = torch.cat([nec_repr, rpb_repr, interaction_mul, interaction_diff], dim=1)  # [B, 4H*D]

        # FC layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        out = self.fc2(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)

        return out

    @staticmethod
    def masked_mean(x, mask):
        # x: [B,T,H], mask: [B,T] (one-hot masks)
        mask = mask.unsqueeze(-1)  # [B,T,1]
        x = x * mask
        lengths = mask.sum(1).clamp(min=1)  # [B,1]
        return x.sum(1) / lengths  # [B,H]

    @staticmethod
    def load_model(filepath, rna_embedding, amino_embedding, **kwargs):
        model = RecommendationModelV2(
            rna_embedding.vocab,
            amino_embedding.vocab,
            **kwargs
        )
        model.load_state_dict(torch.load(filepath))
        return model
