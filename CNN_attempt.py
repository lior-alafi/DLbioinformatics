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

class RNAMinoCNN(nn.Module):
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
        self.expected_score_dim = expected_score_dim

        # Define the 1D Convolutional Layer
        # It takes the combined channels and applies `num_filters` kernels.
        # padding='same' ensures the output length is the same as input length,
        # which simplifies global pooling later.
        self.rna_conv1 = nn.Conv1d(
            in_channels=rna_input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'  # 'same' padding to maintain sequence length
        )
        self.aa_conv1 = nn.Conv1d(
            in_channels=amino_input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same'  # 'same' padding to maintain sequence length
        )
        # Batch Normalization after convolution can help stabilize training
        self.rna_bn1 = nn.BatchNorm1d(num_filters)
        self.aa_bn1 = nn.BatchNorm1d(num_filters)
        self.rna_pool1 = nn.MaxPool1d(2,2)
        self.aa_pool1 = nn.MaxPool1d(2,2)
        # Dropout layer for regularization to prevent overfitting
        # self.dropout = nn.Dropout(dropout_rate)
        self.rna_conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters*2,
            kernel_size=kernel_size,
            padding='same'  # 'same' padding to maintain sequence length
        )
        self.aa_conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters*2,
            kernel_size=kernel_size,
            padding='same'  # 'same' padding to maintain sequence length
        )

        self.rna_bn2 = nn.BatchNorm1d(num_filters*2)
        self.aa_bn2 = nn.BatchNorm1d(num_filters*2)
        self.rna_pool2 = nn.MaxPool1d(2,2)
        self.aa_pool2 = nn.MaxPool1d(2,2)
        # Global Average Pooling layer
        # This layer reduces the spatial dimension (length) to 1,
        # effectively taking the average of features across the entire sequence.
        # Output shape will be (batch_size, num_filters, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.comb1 = nn.Conv1d(1,1,64,3,padding='valid')
        self.comb2 = nn.Conv1d(1,1,128,3,padding='valid')

        self.global_avg_pool2 = nn.AdaptiveAvgPool1d(1)
        # Fully Connected Layer (Dense Layer)
        # It maps the `num_filters` (features from pooling) to the `expected_score_dim`.

        self.fcr = nn.Linear(num_filters*2,num_filters)


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
        # Ensure that the amino_input's sequence length matches rna_input's sequence length
        # for concatenation. This is a critical assumption. If lengths can differ,
        # you might need separate Conv layers for each and then concatenate their pooled outputs.
        # For this implementation, we assume they are padded/truncated to the same effective length
        # or that the amino_input is resized/truncated to match the RNA length for concatenation.
        # A more robust solution might involve separate CNN branches before concatenation.
        # For now, we'll assume the lengths are compatible or amino_input needs to be adjusted.

        # If amino_input and rna_input have different sequence lengths (last dimension),
        # direct concatenation along channels won't work.
        # A common approach is to max-pool or average-pool one to match the other,
        # or use separate convolutional branches and then concatenate their pooled features.
        # For simplicity and based on "stack them on top of each other",
        # we'll assume the lengths are either the same or amino needs to be adjusted.
        # Let's assume for now that `rna_input.shape[2]` and `amino_input.shape[2]`
        # are compatible (e.g., both are `max_len` from CustomDataset).
        # If they are not, you would need to add logic here to align their lengths.
        # For example, by resizing `amino_input` to match `rna_input`'s length.

        # Example of length adjustment (if needed, uncomment and adapt):
        # if rna_input.shape[2] != amino_input.shape[2]:
        #     # Option 1: Resize amino_input to match rna_input length
        #     # This might involve interpolation or cropping, depending on desired behavior.
        #     # For simplicity, let's assume rna_input.shape[2] is the target length.
        #     # This is a placeholder and might need more sophisticated handling.
        #     amino_input = F.interpolate(amino_input, size=rna_input.shape[2], mode='linear', align_corners=False)
        #     print("Warning: Amino input length adjusted to match RNA input length for concatenation.")

        # Stack the RNA and Amino tensors along the channel dimension (dim=1)
        # This effectively combines their features into a single input for the Conv1D layer.
        # Input shape to conv1: (batch_size, rna_channels + amino_channels, sequence_length)
        rna = self.rna_conv1(rna_input)
        rna = self.rna_bn1(rna)
        rna = F.relu(rna)
        rna = self.rna_pool1(rna)
        rna = self.rna_conv2(rna)
        rna = self.rna_bn2(rna)
        rna = F.relu(rna)
        rna = self.rna_pool2(rna)


        aa = self.aa_conv1(amino_input)
        aa = self.aa_bn1(aa)
        aa = F.relu(aa)
        aa = self.aa_pool1(aa)
        aa = self.aa_conv2(amino_input)
        aa = self.aa_bn2(aa)
        aa = F.relu(aa)
        aa = self.aa_pool2(aa)
        x = torch.cat((rna, aa), dim=1)
        # x = self.dropout(x)  # Apply Dropout



        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)  # or x.squeeze(-1)

        x = self.fcr(x)
        output = self.fc(x)

        return output


# --- Example Usage (requires CustomDataset and Preprocessing classes) ---
if __name__ == "__main__":

    rna_vocab_size = 5  # e.g., A, G, C, U, <PAD>
    amino_vocab_size = 21  # e.g., 20 amino acids + <PAD>
    BATCH_SIZE = 2
    # If using n_gram_size > 1 for OneHot, adjust input_channels accordingly:
    # rna_n_gram_size = 3
    # rna_input_channels = rna_vocab_size * rna_n_gram_size
    rna_input_channels = rna_vocab_size
    amino_input_channels = amino_vocab_size

    # Max lengths as defined in CustomDataset
    rna_max_len = 41
    amino_max_len = 912

    expected_score_dim = 200  # As defined in CustomDataset
    rna_prep_emb_1 = OneHotPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=1)
    amino_prep_emb_1 = OneHotPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=1)

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # Instantiate the model
    model = RNAMinoCNN(
        rna_input_channels=rna_input_channels,
        amino_input_channels=amino_input_channels,
        expected_score_dim=1,
        num_filters=64,
        kernel_size=5
    )

    EPOCHS = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
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