import  torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
from accelerate.test_utils.scripts.test_distributed_data_loop import BATCH_SIZE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader3 import CustomDataset, EmbeddingPreprocessing
# from data_loader2 import CustomDataset
from model.lior_model import RecommendationModel
from scipy.stats import pearsonr,spearmanr
BATCH_SIZE = 32
rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=3)
amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=3)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



EPOCHS = 50
model = RecommendationModel(rna_prep_emb_1.vocab, amino_prep_emb_1.vocab, hidden_size=64, output_size=1
                            ,rna_n_gram_size=3,amino_n_gram_size=3)
model.to(device)
# model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        optimizer.step()
    losses.append(np.average(curr_losses))
    y_orig = np.array([])
    y_hats = np.array([])
    for i, ((rna_batch, amino_batch), scores_batch) in tqdm(enumerate(val_loader)):

        rna_batch = rna_batch.to(device)  # Add batch dimension
        amino_batch = amino_batch.to(device)  # Add batch dimension
        outputs = model(rna_batch, amino_batch).to(device)

        # Assuming the target is the RBP index
        y_orig= np.append(y_orig,scores_batch.squeeze().detach().cpu().numpy())
        y_hats = np.append(y_hats,outputs.squeeze().detach().cpu().numpy())

    pear_corr, _ = pearsonr(y_orig, y_hats)
    spear_corr, _ = spearmanr(y_orig, y_hats)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f} pearson: {pear_corr} spearman: {spear_corr}")
# torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
print(losses)