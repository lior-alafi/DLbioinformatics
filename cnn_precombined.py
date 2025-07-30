import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader3 import OneHotPreprocessing, CustomDataset
from lstm_recs2 import BATCH_SIZE
from usage import batch_size


# It's assumed that BasePreprocessing, OneHotPreprocessing, and CustomDataset
# from the previous context are available and correctly configured to produce
# (channels, length) tensors for single samples, which become
# (batch_size, channels, length) when passed in batches.

class RNAMinoCNN1(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) designed to process
    both RNA and Amino acid sequences.

    The model concatenates the processed RNA and Amino tensors along the
    channel dimension, applies convolutional layers, performs global average pooling,
    and outputs a score.

    Args:
        rna_input_channels (int): Number of input channels for RNA sequences.
                                  This will be vocab_size (for n_gram_size=1 OneHot)
                                  or vocab_size * n_gram_size (for n_gram_size > 1 OneHot).
        amino_input_channels (int): Number of input channels for Amino sequences.
                                    Similar to rna_input_channels.
        expected_score_dim (int): The dimension of the output score tensor.
                                  This should match the dimension of your target scores.
        num_filters (int): The number of filters (kernels) to use in the convolutional layers.
                           Defaults to 64 as per the request.
        kernel_size (int): The size of the convolutional kernel.
                           A common choice is 3 or 5 for sequence data.
        dropout_rate (float): Dropout probability for regularization.
    """

    def __init__(self, rna_input_channels: int, amino_input_channels: int,
                 expected_score_dim: int, num_filters: int = 64,
                 kernel_size: int = 5, dropout_rate: float = 0.5):
        super(RNAMinoCNN, self).__init__()

        # Total input channels after concatenating RNA and Amino features
        self.total_input_channels = rna_input_channels + amino_input_channels
        self.expected_score_dim = expected_score_dim

        # Define the 1D Convolutional Layer
        # It takes the combined channels and applies `num_filters` kernels.
        # padding='same' ensures the output length is the same as input length,
        # which simplifies global pooling later.
        self.conv1 = nn.Conv1d(
            in_channels=self.total_input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'  # 'same' padding to maintain sequence length
        )

        # Batch Normalization after convolution can help stabilize training
        self.bn1 = nn.BatchNorm1d(num_filters)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Global Average Pooling layer
        # This layer reduces the spatial dimension (length) to 1,
        # effectively taking the average of features across the entire sequence.
        # Output shape will be (batch_size, num_filters, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer (Dense Layer)
        # It maps the `num_filters` (features from pooling) to the `expected_score_dim`.
        self.fc = nn.Linear(num_filters, self.expected_score_dim)

    def forward(self, rna_input: torch.Tensor, amino_input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the RNAMinoCNN.

        Args:
            rna_input (torch.Tensor): Preprocessed RNA sequence tensor.
                                      Expected shape: (batch_size, rna_input_channels, rna_max_len)
            amino_input (torch.Tensor): Preprocessed Amino sequence tensor.
                                        Expected shape: (batch_size, amino_input_channels, amino_max_len)

        Returns:
            torch.Tensor: The predicted score tensor.
                          Expected shape: (batch_size, expected_score_dim)
        """

        combined_input = torch.cat((rna_input, amino_input), dim=1)
        x = self.conv1(combined_input)
        x = self.bn1(x)  # Apply Batch Normalization
        x = F.relu(x)  # Apply ReLU activation function
        x = self.dropout(x)  # Apply Dropout

        # Output shape: (batch_size, num_filters, 1)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)  # or x.squeeze(-1)
        output = self.fc(x)

        return output


# --- Example Usage (requires CustomDataset and Preprocessing classes) ---
if __name__ == "__main__":

    rna_vocab_size = 5  # e.g., A, G, C, U, <PAD>
    amino_vocab_size = 21  # e.g., 20 amino acids + <PAD>
    BATCH_SIZE = 32

    rna_input_channels = rna_vocab_size
    amino_input_channels = amino_vocab_size

    # Max lengths as defined in CustomDataset
    rna_max_len = 41
    amino_max_len = 912

    effective_sequence_length = rna_max_len

    expected_score_dim = 200  # As defined in CustomDataset
    rna_prep_emb_1 = OneHotPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=3)
    amino_prep_emb_1 = OneHotPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=3)

    # Example usage:
    train_dataset = CustomDataset('data/train_rna_seq.50000.txt',
                                  'data/train_rbps2_seq.50.txt',
                                  'data/train_scores.50000.txt',
                                  rna_max_len=41,
                                  amino_max_len=912,
                                  preprocessing_rna=rna_prep_emb_1,
                                  preprocessing_amino=amino_prep_emb_1)
    test_dataset = CustomDataset('data/validation_rna_seq.12500.txt',
                                 'data/validation_rbps2_seq.13.txt',
                                 'data/validation_scores.12500.txt',
                                 rna_max_len=41,
                                 amino_max_len=912,
                                 preprocessing_rna=rna_prep_emb_1,
                                 preprocessing_amino=amino_prep_emb_1,

                                 )

    train_loader = DataLoader(train_dataset, batch_size=ba, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # Instantiate the model
    model = RNAMinoCNN(
        rna_input_channels=rna_input_channels,
        amino_input_channels=amino_input_channels,
        expected_score_dim=expected_score_dim,
        num_filters=64,
        kernel_size=5
    )

    EPOCHS = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10)

    print(device)

    losses = []

    for epoch in range(EPOCHS):
        curr_losses = []
        y_orig = []
        y_hats = []
        for i, ((rna_batch, amino_batch), scores_batch) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            rna_batch = rna_batch.to(device)  # Add batch dimension
            amino_batch = amino_batch.to(device)  # Add batch dimension

            outputs = model(rna_batch, amino_batch)
            # Assuming the target is the RBP index
            targets = scores_batch.view(-1, 1).to(device)

            loss = criterion(outputs, targets)
            curr_losses.append(loss.item())
            y_orig.append(targets.squeeze().detach().cpu().numpy())
            y_hats.append(outputs.squeeze().detach().cpu().numpy())
            loss.backward()
            # optimizer.step()
            scheduler.step(loss)
        losses.append(np.average(curr_losses))
        y_orig = np.array([])
        y_hats = np.array([])
        for i, ((rna_batch, amino_batch), scores_batch) in tqdm(enumerate(val_loader)):
            rna_batch = rna_batch.to(device)  # Add batch dimension
            amino_batch = amino_batch.to(device)  # Add batch dimension
            outputs = model(rna_batch, amino_batch).to(device)

            # Assuming the target is the RBP index
            y_orig = np.append(y_orig, scores_batch.squeeze().detach().cpu().numpy())
            y_hats = np.append(y_hats, outputs.squeeze().detach().cpu().numpy())

        pear_corr, _ = pearsonr(y_orig, y_hats)
        spear_corr, _ = spearmanr(y_orig, y_hats)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f} pearson: {pear_corr} spearman: {spear_corr}")
    # torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
    print(losses)