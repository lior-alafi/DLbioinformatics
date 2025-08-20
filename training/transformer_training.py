import random
import torch.nn as nn
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader4 import PreprocessingSharedEmbedding, CustomDataset
from model.lior_transformer import TransformerEncoderModel
from model.lior_transformer2 import TransformerEncoderModelV2
from utils.experimental_tracker import ExperimentTracker
from utils.metrics import metrics, pearson_evaluation
from utils.tripletloss import TripletLoss
all_scores = np.loadtxt('../data/train_scores.102578.txt', delimiter='\t').flatten().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(all_scores)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
shared_pre = PreprocessingSharedEmbedding(rna_file='../data/train_rna_seq.102578.txt', rbp_file='../data/train_rbps2_seq.170.txt', n_gram_size=3, step_size=1)
train_dataset = CustomDataset('../data/train_rna_seq.102578.txt',
                              'data/train_rbps2_seq.170.txt',
                              'data/train_scores.102578.txt',
                              rna_max_len=41,
                              amino_max_len=912,
                              preprocessing_rna=shared_pre, preprocessing_amino=shared_pre, expected_score_dim=170, score_transform=scaler)
# Example usage:

test_dataset = CustomDataset('../data/validation_rna_seq.18100.txt',
                             'data/validation_rbps2_seq.30.txt',
                            'data/validation_scores.18100.txt', expected_score_dim=30,
                             rna_max_len=41,
                             amino_max_len=912,
                             preprocessing_rna=shared_pre,
                             preprocessing_amino=shared_pre,
                             score_transform=scaler
                             )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
embed_options = [32,64,128,256]
hidden_options = [32,64,128,256]
lstm_l_options = [1,2,3,4,5,6,7,8]
heads_options = [1,2,4,8]

EPOCHS = 10
for experiment in range(1):
    EMBED_DIM = 64 #144 #random.choice(embed_options)
    HIDDEN_SIZE =128 #768 #random.choice(hidden_options)
    LSTM_LAYER = 2 #11 #random.choice(lstm_l_options)
    NUM_HEADS= 4 #12# random.choice(heads_options)
    # alpha_choices = [0.05, 0.1, 0.2]
    # beta_choices = [0.0, 0.05, 0.1]
    # gamma_choices = [0.5, 1.0, 1.5, 2.0]
    # ls = (
    #     random.choice(alpha_choices),
    #     random.choice(beta_choices),
    #     random.choice(gamma_choices)
    # )

    # Bidirectional= random.choice([True,False])
    # fc= random.choice([[64],[128],[256],[512],[1024],[2048],[4096]])

    LR= 0.001# 2e-05 #



    model = TransformerEncoderModelV2(shared_pre.vocab_size,embedding_dim=EMBED_DIM,num_layers=LSTM_LAYER,
                                    num_heads=NUM_HEADS,hidden_dim=HIDDEN_SIZE,dropout=0.5,output_dim=1
                                 )
    model.to(device)

    hyperparams = {
        "embed_dim": EMBED_DIM,
        "hidden_size": HIDDEN_SIZE,
        "layers": LSTM_LAYER,
        "heads": NUM_HEADS,
        # "alpha": ls[0],
        # "beta": ls[1],
        # "gamma": ls[2]
    }
    tracker = ExperimentTracker(
        out_dir="../experiments_out",
        exp_name=f"Transformer_{experiment}",
        hyperparams=hyperparams,
        track_best_by="pearson"
    )

    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = nn.L1Loss() #TripletLoss(alpha=ls[0],beta=ls[1],gamma=ls[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',patience=10)
    description = f'Trasformer encoder epochs{EPOCHS} hidden {HIDDEN_SIZE} embed_dim {EMBED_DIM} Layers {LSTM_LAYER} heads {NUM_HEADS} loss L1'
    print(device)
    print(description)

    losses = []
    val_loss = []
    val_pearson= []
    val_spearman=[]
    bad_pearson_count = 0

    for epoch in range(EPOCHS):
        model.train()
        curr_losses = []


        for i, (( rna_batch, rbp_batch, rna_mask, rbp_mask), targets) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            rna_batch = rna_batch.to(device)
            rbp_batch = rbp_batch.to(device)
            rna_mask = rna_mask.to(device)
            rbp_mask = rbp_mask.to(device)
            targets = targets.to(device)

            outputs = model(rna_batch, rbp_batch, rna_mask, rbp_mask).view(-1)
            targets = targets.to(device).view(-1)


            loss = criterion(outputs, targets)
            if targets.shape != outputs.shape:
                print(f"targets shape: {targets.shape}, output shape: {outputs.shape}")

            loss.backward()
            optimizer.step()

            curr_losses.append(loss.item())

        losses.append(np.mean(curr_losses))

        model.eval()
        with torch.no_grad():
            curr_losses = []
            y_orig = []
            y_hats = []

            for i, ((rna_batch, rbp_batch, rna_mask, rbp_mask), scores_batch) in tqdm(enumerate(val_loader)):
                rna_batch = rna_batch.to(device)
                rbp_batch = rbp_batch.to(device)
                rna_mask = rna_mask.to(device)
                rbp_mask = rbp_mask.to(device)
                targets = scores_batch.view(-1).to(device)

                outputs = model(rna_batch, rbp_batch, rna_mask, rbp_mask).view(-1)

                loss1 = criterion(outputs, targets)  # Loss משולב
                curr_losses.append(loss1.item())

                y_orig.extend(targets.cpu().numpy().ravel())
                y_hats.extend(outputs.cpu().numpy().ravel())

            val_loss_epoch = np.mean(curr_losses)
            val_loss.append(val_loss_epoch)
            # scheduler.step(val_loss_epoch)

            # כאן חישוב Pearson/Spearman
            bad_pearson_count, pear_corr, spear_corr = pearson_evaluation(
                y_orig, y_hats, bad_pearson_count, epoch,
                scaler, use_inverse=False
            )
            val_pearson.append(pear_corr)
            val_spearman.append(spear_corr)
            tracker.log_epoch(epoch, losses[-1], val_loss_epoch, pear_corr, spear_corr, model)

            print(f"Epoch [{epoch + 1}/{EPOCHS}], train_Loss: {losses[-1]:.4f}, val_loss: {val_loss_epoch:.4f}, pearson: {val_pearson[-1]:.4f}, spearman: {val_spearman[-1]:.4f}")

            # if bad_pearson_count >= 2:
            #         print("⛔ Early stopping: Pearson undefined for 2 consecutive epochs.")
            #         break



    # torch.save(model.state_dict(), 'model/saved_models/lior2_8_7_25.pt')

    print(losses)
    print(description)
    metrics(description,losses,val_loss,val_pearson,val_spearman)
    tracker.export()