import  torch
import torch.nn as nn
import random

import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader4 import CustomDataset, EmbeddingPreprocessing,EmbeddingKMERPreprocessing
from scipy.stats import pearsonr,spearmanr

from model.lior_model_lstm2 import RecommendationModelV2
from utils.metrics import metrics
all_scores = np.loadtxt('data/train_scores.102578.txt', delimiter='\t').flatten().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(all_scores)
BATCH_SIZE = 128
# rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=3)
# amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=3)
rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.102578.txt', n_gram_size=1)
amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.170.txt', n_gram_size=1)

# Example usage:
train_dataset = CustomDataset('data/train_rna_seq.102578.txt',
                              'data/train_rbps2_seq.170.txt',
                              'data/train_scores.102578.txt',
                              rna_max_len=41,
                              amino_max_len=912,
                              preprocessing_rna=rna_prep_emb_1,
                              preprocessing_amino=amino_prep_emb_1,expected_score_dim=170,score_transform=scaler)
test_dataset = CustomDataset('data/validation_rna_seq.18100.txt',
                             'data/validation_rbps2_seq.30.txt',
                            'data/validation_scores.18100.txt',expected_score_dim=30,
                             rna_max_len=41,
                             amino_max_len=912,
                             score_transform=scaler,
preprocessing_rna=rna_prep_emb_1,
                              preprocessing_amino=amino_prep_emb_1,

                             )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_options = [8,16,32,64,128,256]
hidden_options = [32,64,128,256]
lstm_l_options = [1,2,4,8,10]
EPOCHS = 30
# EMBED_DIM = 128
# HIDDEN_SIZE = 128
# LSTM_LAYER = 4
# LR= 0.01
def training_loop(train_loader, device, model, criterion, optimizer, curr_losses):
    for i, ((rna_batch, rbp_batch, _, _), scores_batch) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        rna_batch = rna_batch.to(device)
        rbp_batch = rbp_batch.to(device)
        targets = scores_batch.view(-1, 1).to(device)

        outputs = model(rna_batch, rbp_batch)
        loss = criterion(outputs, targets)
        if targets.shape != outputs.shape:
            print(f"score shape: {scores_batch.shape}, output shape: {outputs.shape}")

        loss.backward()
        optimizer.step()

        curr_losses.append(loss.item())



for experiment in range(10):
    EMBED_DIM = random.choice(embed_options)
    HIDDEN_SIZE = random.choice(hidden_options)
    LSTM_LAYER =  random.choice(lstm_l_options)
    Bidirectional= random.choice([True,False])
    fc= random.choice([[64],[128],[256],[512],[1024],[2048],[4096]])
    bad_pearson_count = 0
    LR= 0.001



    model = RecommendationModelV2(rna_prep_emb_1.vocab, amino_prep_emb_1.vocab,embedding_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=1
                                ,rna_n_gram_size=1,amino_n_gram_size=1,lstm_layers=LSTM_LAYER,
                                  bidirectional=Bidirectional,fc_units=fc,
                                )
    model.to(device)


    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',patience=10)

    print(device)
    description = f'LSTM epochs:{EPOCHS} hidden: {HIDDEN_SIZE}, embed_dim: {EMBED_DIM} Layers: {LSTM_LAYER} bidirectional {Bidirectional} fc {fc}'
    print(description)

    losses = []
    val_loss = []
    val_pearson= []
    val_spearman=[]

    for epoch in range(EPOCHS):
        model.train()
        curr_losses = []
        if bad_pearson_count >= 2:
            print("⛔ Early stopping: Pearson undefined due to low std in 2 consecutive epochs.")
            break
        training_loop(train_loader, device, model, criterion, optimizer, curr_losses)

        losses.append(np.mean(curr_losses))

        model.eval()
        with torch.no_grad():
            curr_losses = []
            y_orig = []
            y_hats = []

            for i, ((rna_batch, amino_batch, _, _), scores_batch) in tqdm(enumerate(val_loader)):
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

            y_orig = np.array(y_orig, dtype=np.float32).reshape(-1, 1)
            y_hats = np.array(y_hats, dtype=np.float32).reshape(-1, 1)
            print(f'y_true {y_orig.shape} y_hat {y_hats.shape}')

            # Unnormalize
            y_orig_real = scaler.inverse_transform(y_orig).flatten()
            y_hats_real = scaler.inverse_transform(y_hats).flatten()
            assert not np.isnan(y_hats_real).any(), "y_hats_real contains NaN"
            assert not np.isinf(y_hats_real).any(), "y_hats contains inf"

            std_y_hat = np.std(y_hats_real)
            std_y_true = np.std(y_orig_real)

            if std_y_hat < 1e-6 or std_y_true < 1e-6:
                print(
                    f"⚠️ Warning (epoch {epoch + 1}): std is near-zero (y_hat: {std_y_hat:.6f}, y_true: {std_y_true:.6f}) – skipping Pearson.")
                pear_corr, spear_corr = 0.0, 0.0
                bad_pearson_count += 1
            else:
                pear_corr, _ = pearsonr(y_orig_real, y_hats_real)
                spear_corr, _ = spearmanr(y_orig_real, y_hats_real)
                bad_pearson_count = 0  # reset if Pearson is valid

            val_pearson.append(pear_corr)
            val_spearman.append(spear_corr)

            print(f"Epoch [{epoch + 1}/{EPOCHS}], train_Loss: {losses[-1]:.4f}, val_loss: {val_loss_epoch:.4f}, pearson: {val_pearson[-1]:.4f}, spearman: {val_spearman[-1]:.4f}")
            if bad_pearson_count > 0:
                print(f"⚠️ bad_pearson_count = {bad_pearson_count} (will stop at 2)")

    # torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
    print(losses)
    print()
    metrics(description,losses,val_loss,val_pearson,val_spearman)