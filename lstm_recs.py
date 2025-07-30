import  torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model.lior_model import RecommendationModel
from shitty_dataloader import TextDataset
from scipy.stats import pearsonr,spearmanr

# Example usage:
train_dataset = TextDataset('data/train_rna_seq.50000.txt', 'data/train_rbps2_seq.50.txt')
test_dataset = TextDataset('data/validation_rna_seq.10000.txt', 'data/validation_rbps2_seq.10.txt')
#dataset size  = 24135600
# rpb_seq, nec_seq, rpb_index, nec_index = dataset[50]
# print(f"RBP Sequence: {rpb_seq}")
# print(f"Nucleotide Sequence: {nec_seq}")
# print(f"RBP Index: {rpb_index}")
# print(f"Nucleotide Index: {nec_index}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # def forward(self, x):
    #     lstm_out, _ = self.lstm(x)
    #     out = self.fc(lstm_out[:, -1, :])  # Use the last output of LSTM
    #     return out

EPOCHS = 10
model = RecommendationModel(train_dataset.rna_mapping, train_dataset.amino_acid_mapping, hidden_size=64, output_size=1)
model.to(device)
# model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
y = []
with open('data/train_scores.50000.txt', 'r') as f:
            for line in f:
                y.append([float(x) for x in line.strip().split()])

y_val = []
with open('data/validation_scores.10000.txt', 'r') as f:
            for line in f:
                y_val.append([float(x) for x in line.strip().split()])

print(device)
print(np.array(y).shape) #(120678, 200)
# Training loop
# 0 - 200000

losses = []

for epoch in range(EPOCHS):
    curr_losses = []
    y_orig = []
    y_hats = []
    for i in tqdm(range(train_dataset.length),'looping through dataset'):
        idxs = np.arange(train_dataset.length)
        np.random.shuffle(idxs)
        rpb_seq, nec_seq, rpb_index, nec_index = train_dataset[idxs[i]]
        optimizer.zero_grad()
        rpb_seq = rpb_seq.unsqueeze(0).to(device)  # Add batch dimension
        nec_seq = nec_seq.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(nec_seq,rpb_seq)
        targets = y[nec_index][rpb_index]  # Get the target value from y

        # Assuming the target is the RBP index
        targets = torch.tensor([targets], dtype=torch.float32).to(device)

        loss = criterion(outputs.squeeze(), targets.squeeze())
        curr_losses.append(loss.item())
        y_orig.append(targets.squeeze().detach().cpu().numpy())
        y_hats.append(outputs.squeeze().detach().cpu().numpy())
        loss.backward()
        optimizer.step()
    losses.append(np.average(curr_losses))
    y_orig = []
    y_hats = []
    for j in range(test_dataset.length):
        rpb_seq, nec_seq, rpb_index, nec_index = test_dataset[j]
        optimizer.zero_grad()
        rpb_seq = rpb_seq.unsqueeze(0).to(device)  # Add batch dimension
        nec_seq = nec_seq.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(rpb_seq, nec_seq).to(device)
        targets = y_val[nec_index][rpb_index]  # Get the target value from y

        # Assuming the target is the RBP index
        targets = torch.tensor([targets], dtype=torch.float32).to(device)
        y_orig.append(targets.squeeze().detach().cpu().numpy())
        y_hats.append(outputs.squeeze().detach().cpu().numpy())
    pear_corr, _ = pearsonr(y_orig, y_hats)
    spear_corr, _ = spearmanr(y_orig, y_hats)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f} pearson: {pear_corr} spearman: {spear_corr}")
torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
print(losses)