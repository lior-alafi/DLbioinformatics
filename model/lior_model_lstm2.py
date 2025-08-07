import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommendationModelV2(nn.Module):
    def __init__(self, nucleotide_vocab, amino_vocab, hidden_size, output_size,
                 rna_n_gram_size=1, amino_n_gram_size=1,
                 embedding_dim=128, fc_units=[256], lstm_layers=1,
                 bidirectional=False):
        super(RecommendationModelV2, self).__init__()
        self.AGCU_V = nucleotide_vocab
        self.AA_V = amino_vocab
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.rna_n_gram_size = rna_n_gram_size
        self.amino_n_gram_size = amino_n_gram_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Embedding layers
        self.AGCU_embedding = nn.Embedding(len(self.AGCU_V), self.embedding_dim)
        self.AA_embedding = nn.Embedding(len(self.AA_V), self.embedding_dim)

        # LSTM input sizes (handle n-grams)
        rna_lstm_input_size = self.embedding_dim if self.rna_n_gram_size <= 1 else self.embedding_dim * self.rna_n_gram_size
        amino_lstm_input_size = self.embedding_dim if self.amino_n_gram_size <= 1 else self.embedding_dim * self.amino_n_gram_size

        # LSTM layers
        self.AGCU_lstm = nn.LSTM(
            input_size=rna_lstm_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        self.AA_lstm = nn.LSTM(
            input_size=amino_lstm_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Fully connected layers
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # Because we use concat(rna, rbp, mul, diff)
        self.final_repr_size = hidden_size * self.num_directions
        self.merge_dim = self.final_repr_size * 4
        self.fc1 = nn.Linear(self.merge_dim, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[0]*2)
        self.fc3 = nn.Linear(fc_units[0]*2, output_size)

    def forward(self, nec_seq, rpb_seq):
        # Handle RBP input
        if self.amino_n_gram_size > 1 and len(rpb_seq.shape) == 4:
            batch_size, max_len, _ = rpb_seq.shape
            rpb_seq_reshaped = rpb_seq.view(-1, self.amino_n_gram_size)
            rpb_embedded_flat = self.AA_embedding(rpb_seq_reshaped)
            rpb_embedded = rpb_embedded_flat.view(batch_size, max_len,
                                                  self.amino_n_gram_size * self.embedding_dim)
        else:
            rpb_embedded = self.AA_embedding(rpb_seq)

        # Handle RNA input
        if self.rna_n_gram_size > 1 and len(nec_seq.shape) == 4:
            batch_size, max_len, _ = nec_seq.shape
            nec_seq_reshaped = nec_seq.view(-1, self.rna_n_gram_size)
            nec_embedded_flat = self.AGCU_embedding(nec_seq_reshaped)
            nec_embedded = nec_embedded_flat.view(batch_size, max_len,
                                                  self.rna_n_gram_size * self.embedding_dim)
        else:
            nec_embedded = self.AGCU_embedding(nec_seq)

        # Pass through LSTMs
        rpb_out, _ = self.AA_lstm(rpb_embedded)   # [B, T, H*D]
        nec_out, _ = self.AGCU_lstm(nec_embedded) # [B, T, H*D]

        # Mean pooling over time steps
        rpb_repr = rpb_out.mean(dim=1)  # [B, H*D]
        nec_repr = nec_out.mean(dim=1)  # [B, H*D]

        # Merge features
        interaction_mul = nec_repr * rpb_repr
        interaction_diff = torch.abs(nec_repr - rpb_repr)
        combined = torch.cat([nec_repr, rpb_repr, interaction_mul, interaction_diff], dim=1)  # [B, 4H*D]

        # FC layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        out = self.fc2(x)  # [B, output_size]
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)

        return out


