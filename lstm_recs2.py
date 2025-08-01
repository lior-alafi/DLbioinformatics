import  torch
import torch.nn as nn
import random

import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader3 import CustomDataset, EmbeddingPreprocessing,EmbeddingKMERPreprocessing
from model.lior_model import RecommendationModel
from scipy.stats import pearsonr,spearmanr

from utils.metrics import metrics

BATCH_SIZE = 128
# rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=3)
# amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=3)
rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=1)
amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=1)

# Example usage:
train_dataset = CustomDataset('data/train_rna_seq.50000.txt',
                              'data/train_rbps2_seq.50.txt',
                              'data/train_scores.50000.txt',
                              rna_max_len=41,
                              amino_max_len=912,
                              preprocessing_rna=rna_prep_emb_1,
                              preprocessing_amino=amino_prep_emb_1,expected_score_dim=50)
test_dataset = CustomDataset('data/validation_rna_seq.12500.txt',
                             'data/validation_rbps2_seq.13.txt',
                            'data/validation_scores.12500.txt',expected_score_dim=13,
                             rna_max_len=41,
                             amino_max_len=912,
preprocessing_rna=rna_prep_emb_1,
                              preprocessing_amino=amino_prep_emb_1,

                             )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_options = [8,16,32,64,128,256]
hidden_options = [32,64,128,256,516]
lstm_l_options = [1,2,4,8,10]
EPOCHS = 50
# EMBED_DIM = 128
# HIDDEN_SIZE = 128
# LSTM_LAYER = 4
# LR= 0.01
for experiment in range(15):
    EMBED_DIM = random.choice(embed_options)
    HIDDEN_SIZE = random.choice(hidden_options)
    LSTM_LAYER = random.choice(lstm_l_options)
    LR= 0.01



    model = RecommendationModel(rna_prep_emb_1.vocab, amino_prep_emb_1.vocab,embedding_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=1
                                ,rna_n_gram_size=1,amino_n_gram_size=1,lstm_layers=LSTM_LAYER)
    model.to(device)


    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',patience=10)

    print(device)
    print(f'LSTM epochs:{EPOCHS} hidden: {HIDDEN_SIZE}, embed_dim: {EMBED_DIM} Layers: {LSTM_LAYER}')

    losses = []
    val_loss = []
    val_pearson= []
    val_spearman=[]

    for epoch in range(EPOCHS):
        model.train()
        curr_losses = []

        for i, ((rna_batch, amino_batch), scores_batch) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            rna_batch = rna_batch.to(device)
            amino_batch = amino_batch.to(device)
            targets = scores_batch.view(-1, 1).to(device)

            outputs = model(rna_batch, amino_batch)
            loss = criterion(outputs, targets)
            if targets.shape != outputs.shape:
                print(f"score shape: {scores_batch.shape}, output shape: {outputs.shape}")

            loss.backward()
            optimizer.step()

            curr_losses.append(loss.item())

        losses.append(np.mean(curr_losses))

        model.eval()
        with torch.no_grad():
            curr_losses = []
            y_orig = []
            y_hats = []

            for i, ((rna_batch, amino_batch), scores_batch) in tqdm(enumerate(val_loader)):
                rna_batch = rna_batch.to(device)
                amino_batch = amino_batch.to(device)
                targets = scores_batch.view(-1, 1).to(device)


                outputs = model(rna_batch, amino_batch)
                # print(f"outputs.shape: {outputs.shape}, targets.shape: {targets.shape}")
                loss1 = criterion(outputs, targets)
                curr_losses.append(loss1.item())

                y_orig.extend(targets.squeeze().cpu().numpy())
                y_hats.extend(outputs.squeeze().cpu().numpy())
            val_loss_epoch = np.mean(curr_losses)
            val_loss.append(val_loss_epoch)
            scheduler.step(val_loss_epoch)
            y_orig = np.array(y_orig,dtype=np.float32)
            y_hats = np.array(y_hats,dtype=np.float32)
            print(f'y_true {y_orig.shape} y_hat {y_hats.shape}')
            pear_corr, _ = pearsonr(y_orig, y_hats)
            spear_corr, _ = spearmanr(y_orig, y_hats)
            val_pearson.append(pear_corr)
            val_spearman.append(spear_corr)

            print(f"Epoch [{epoch + 1}/{EPOCHS}], train_Loss: {losses[-1]:.4f}, val_loss: {val_loss_epoch:.4f}, pearson: {val_pearson[-1]:.4f}, spearman: {val_spearman[-1]:.4f}")
    # torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
    print(losses)
    metrics(losses,val_loss,val_pearson,val_spearman,HIDDEN_SIZE,LSTM_LAYER,LR,EPOCHS,EMBED_DIM)