import torch
from torch import nn
import torch.nn.functional as F
#8_7_25
class RecommendationModel(nn.Module):
    def __init__(self, nucleotide_vocab, amino_vocab, hidden_size, output_size,rna_n_gram_size=1,amino_n_gram_size=1):
        super(RecommendationModel, self).__init__()
        self.AGCU_V = nucleotide_vocab
        self.AA_V = amino_vocab
        self.hidden_size = hidden_size
        self.rna_n_gram_size = rna_n_gram_size
        self.amino_n_gram_size = amino_n_gram_size
        self.output_size = output_size
        self.AGCU_embedding = nn.Embedding(len(self.AGCU_V), 128)
        self.AA_embedding = nn.Embedding(len(self.AA_V), 128)
        rna_lstm_input_size = 128 if self.rna_n_gram_size <= 1 else 128*self.rna_n_gram_size
        amino_n_gram_size= 128 if self.amino_n_gram_size <= 1 else 128*self.amino_n_gram_size
        self.AGCU_lstm = nn.LSTM(rna_lstm_input_size, hidden_size, batch_first=True)
        self.AA_lstm = nn.LSTM(amino_n_gram_size, hidden_size, batch_first=True)
        # Concatenate the outputs of both LSTMs
        # and pass through a fully connected layer

        self.concat_size = hidden_size * 2  # Two LSTM outputs
        self.fc1 = nn.Linear(self.concat_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self,nec_seq, rpb_seq):
        """
        Forward pass of the model.
        """
        """
        Forward pass of the model.
        """
        # Embed the sequences
        if self.amino_n_gram_size > 1:
            # Reshape from (batch_size, max_len, n_gram_size) to (batch_size * max_len, n_gram_size)
            # to apply embedding correctly
            batch_size, max_len, _ = rpb_seq.shape
            rpb_seq_reshaped = rpb_seq.view(-1, self.amino_n_gram_size)  # (batch_size * max_len, n_gram_size)

            # Embed each index in the n-gram window
            rpb_embedded_flat = self.AA_embedding(rpb_seq_reshaped)  # (batch_size * max_len, n_gram_size, 128)

            # Concatenate the embeddings along the last dimension to form a single feature vector for each window
            rpb_embedded = rpb_embedded_flat.view(batch_size, max_len,
                                                  self.amino_n_gram_size * 128)  # (batch_size, max_len, n_gram_size * 128)
        else:
            rpb_embedded = self.AA_embedding(rpb_seq)  # (batch_size, max_len, 128)

            # Handle n_gram_size > 1 for nec_seq (Nucleotides) - similar logic
        if self.rna_n_gram_size > 1:
            batch_size, max_len, _ = nec_seq.shape
            nec_seq_reshaped = nec_seq.view(-1, self.rna_n_gram_size)  # (batch_size * max_len, n_gram_size)

            nec_embedded_flat = self.AGCU_embedding(nec_seq_reshaped)  # (batch_size * max_len, n_gram_size, 128)

            nec_embedded = nec_embedded_flat.view(batch_size, max_len,
                                                  self.rna_n_gram_size * 128)  # (batch_size, max_len, n_gram_size * 128)
        else:
            nec_embedded = self.AGCU_embedding(nec_seq)  # (batch_size, max_len, 128)

        # Pass through LSTM layers
        rpb_out, _ = self.AA_lstm(rpb_embedded)
        nec_out, _ = self.AGCU_lstm(nec_embedded)

        # Use the last output of LSTM for each sequence
        rpb_out = rpb_out[:, -1, :]
        nec_out = nec_out[:, -1, :]

        # Concatenate the outputs
        combined = torch.cat((rpb_out, nec_out), dim=1)

        # Fully connected layers
        out = F.relu(self.fc1(combined))
        out = self.fc2(out)

        return out
    # docker
    # run - -gpus = all - p
    # 127.0
    # .0
    # .1: 9000:8080
    # us - docker.pkg.dev / colab - images / public / runtime