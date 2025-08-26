import  torch
import torch.nn as nn
import random

import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
from scipy.stats import pearsonr,spearmanr

from model.lior_model_lstm2 import RecommendationModelV2
from utils.dataloader4 import EmbeddingPreprocessing, CustomDataset
from utils.metrics import metrics, pearson_evaluation
from utils.tripletloss import TripletLoss

all_scores = np.loadtxt('../data/train_scores.102578.txt', delimiter='\t').flatten().reshape(-1, 1)
# scaler = StandardScaler()
scaler = joblib.load("../model/final/scaler.joblib")
scaler.fit(all_scores)
#save scaler
# joblib.dump(scaler, "model/final/scaler.joblib")

BATCH_SIZE = 64
# rna_prep_emb_1 = EmbeddingPreprocessing('data/train_rna_seq.50000.txt', n_gram_size=3)
# amino_prep_emb_1 = EmbeddingPreprocessing('data/train_rbps2_seq.50.txt', n_gram_size=3)
rna_prep_emb_1 = EmbeddingPreprocessing('../data/train_rna_seq.102578.txt', n_gram_size=1)
amino_prep_emb_1 = EmbeddingPreprocessing('../data/train_rbps2_seq.170.txt', n_gram_size=1)
rna_prep_emb_1.save('../model/final/rna_preproccessor.pt')
amino_prep_emb_1.save('../model/final/amino_preproccessor.pt')
# Example usage:
train_dataset = CustomDataset('../data/train_rna_seq.102578.txt',
                              '../data/train_rbps2_seq.170.txt',
                              '../data/train_scores.102578.txt',
                              rna_max_len=41,
                              amino_max_len=912,
                              preprocessing_rna=rna_prep_emb_1,
                              preprocessing_amino=amino_prep_emb_1, expected_score_dim=170, score_transform=scaler)
test_dataset = CustomDataset('../data/validation_rna_seq.18100.txt',
                             '../data/validation_rbps2_seq.30.txt',
                            '../data/validation_scores.18100.txt', expected_score_dim=30,
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
lstm_l_options = [1,2,4,8]
EPOCHS = 10
# EMBED_DIM = 128
# HIDDEN_SIZE = 128
# LSTM_LAYER = 4
# LR= 0.01
def training_loop(train_loader, device, model, criterion, optimizer, curr_losses):
    for i, ((rna_batch, rbp_batch,rna_mask, rbp_mask), scores_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        rna_batch = rna_batch.to(device)
        rbp_batch = rbp_batch.to(device)
        rna_mask = rna_mask.to(device)
        rbp_mask = rbp_mask.to(device)

        targets = scores_batch.view(-1, 1).to(device)

        outputs = model(rna_batch, rbp_batch,rna_mask,rbp_mask)
        loss = criterion(outputs, targets)
        if targets.shape != outputs.shape:
            print(f"score shape: {scores_batch.shape}, output shape: {outputs.shape}")

        loss.backward()
        optimizer.step()

        curr_losses.append(loss.item())





for experiment in range(3):
    EMBED_DIM = 64 # random.choice(embed_options)
    HIDDEN_SIZE = 128 # random.choice(hidden_options)
    LSTM_LAYER = 2 #  random.choice(lstm_l_options)
    Bidirectional= True # random.choice([True,False])
    fc= [64]  #random.choice([[64],[128],[256],[512]])

    # alpha = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
    # beta = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
    # gamma = random.choice([0.0,0.25,0.5,0.75,1.0,1.25])
    bad_pearson_count = 0
    LR= 0.001



    model = RecommendationModelV2(rna_prep_emb_1.vocab, amino_prep_emb_1.vocab,embedding_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=1
                                ,rna_n_gram_size=1,amino_n_gram_size=1,lstm_layers=LSTM_LAYER,
                                  bidirectional=Bidirectional,fc_units=fc,
                                )
    model.to(device)

    if experiment == 0:
        EPOCHS = 5
        model.load_state_dict(torch.load('../model/final/LSTM_V2_epochs_10_hidden_128_embed_dim_64_Layers_2_bidirectional_True_fc_[64]_MSE__pearson_0.26414856735218467_exp0_epoch_5.pt'))
    else:
        EPOCHS = 10
    criterion = nn.MSELoss() #TripletLoss(alpha=alpha,beta=beta,gamma=gamma)#nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',patience=10)

    print(device)
    description = f'LSTM V2 epochs {EPOCHS} hidden {HIDDEN_SIZE} embed_dim {EMBED_DIM} Layers {LSTM_LAYER} bidirectional {Bidirectional} fc {fc} MSE'
    print(description)

    losses = []
    val_loss = []
    val_pearson= []
    val_spearman=[]
    best_pearson = 0.0
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

            for i, ((rna_batch, amino_batch,rna_mask, rbp_mask), scores_batch) in enumerate(tqdm(val_loader)):
                rna_batch = rna_batch.to(device)
                amino_batch = amino_batch.to(device)
                rna_mask = rna_mask.to(device)
                rbp_mask = rbp_mask.to(device)
                targets = scores_batch.view(-1, 1).to(device)

                outputs = model(rna_batch, amino_batch, rna_mask, rbp_mask)
                # print(f"outputs.shape: {outputs.shape}, targets.shape: {targets.shape}")
                loss1 = criterion(outputs, targets)
                curr_losses.append(loss1.item())

                y_orig.extend(targets.squeeze().cpu().numpy())
                y_hats.extend(outputs.squeeze().cpu().numpy())
            val_loss_epoch = np.mean(curr_losses)
            val_loss.append(val_loss_epoch)
            scheduler.step(val_loss_epoch)

            bad_pearson_count,pear_corr,spear_corr = pearson_evaluation(y_orig,y_hats, bad_pearson_count,epoch,scaler)
            val_pearson.append(pear_corr)
            val_spearman.append(spear_corr)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], train_Loss: {losses[-1]:.4f}, val_loss: {val_loss_epoch:.4f}, pearson: {val_pearson[-1]:.4f}, spearman: {val_spearman[-1]:.4f}")
            if pear_corr < 0.000000:
                print('negative pearson')
                break

            print(f'saving model')
            torch.save(model.state_dict(), f'../model/final/{description.replace(' ','_')}__pearson_{val_pearson[-1]}_exp{experiment}_epoch_{epoch}.pt')
            if bad_pearson_count > 0:
                print(f"⚠️ bad_pearson_count = {bad_pearson_count} (will stop at 2)")

    # torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')
    print(losses)
    print()
    metrics(description,losses,val_loss,val_pearson,val_spearman)