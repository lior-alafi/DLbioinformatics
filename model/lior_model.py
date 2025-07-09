import torch
from torch import nn
import torch.nn.functional as F
#8_7_25
class RecommendationModel(nn.Module):
    def __init__(self, nucleotide_vocab, amino_vocab, hidden_size, output_size):
        super(RecommendationModel, self).__init__()
        self.AGCU_V = nucleotide_vocab
        self.AA_V = amino_vocab
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.AGCU_embedding = nn.Embedding(len(self.AGCU_V), 128)
        self.AA_embedding = nn.Embedding(len(self.AA_V), 128)
        self.AGCU_lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.AA_lstm = nn.LSTM(128, hidden_size, batch_first=True)
        # Concatenate the outputs of both LSTMs
        # and pass through a fully connected layer

        self.concat_size = hidden_size * 2  # Two LSTM outputs
        self.fc1 = nn.Linear(self.concat_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, rpb_seq, nec_seq):
        """
        Forward pass of the model.
        """
        # Embed the sequences
        rpb_embedded = self.AA_embedding(rpb_seq)
        nec_embedded = self.AGCU_embedding(nec_seq)

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

        return

    # docker
    # run - -gpus = all - p
    # 127.0
    # .0
    # .1: 9000:8080
    # us - docker.pkg.dev / colab - images / public / runtime