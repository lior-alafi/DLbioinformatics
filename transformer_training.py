import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader4 import PreprocessingSharedEmbedding, CustomDataset
from model.lior_transformer import TransformerEncoderModel
from utils.metrics import metrics
from utils.tripletloss import TripletLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
shared_pre = PreprocessingSharedEmbedding(rna_file='data/train_rna_seq.102578.txt', rbp_file='data/train_rbps2_seq.170.txt', n_gram_size=3, step_size=1)
train_dataset = CustomDataset('data/train_rna_seq.102578.txt',
                              'data/train_rbps2_seq.170.txt',
                              'data/train_scores.102578.txt',
                              rna_max_len=41,
                              amino_max_len=912,
                             preprocessing_rna=shared_pre, preprocessing_amino=shared_pre,expected_score_dim=170)
# Example usage:

test_dataset = CustomDataset('data/validation_rna_seq.18100.txt',
                             'data/validation_rbps2_seq.30.txt',
                            'data/validation_scores.18100.txt',expected_score_dim=30,
                             rna_max_len=41,
                             amino_max_len=912,
preprocessing_rna=shared_pre,
                              preprocessing_amino=shared_pre,

                             )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

EPOCHS = 10
for experiment in range(1):
    EMBED_DIM = 128 #random.choice(embed_options)
    HIDDEN_SIZE =256  #random.choice(hidden_options)
    LSTM_LAYER =  2 # random.choice(lstm_l_options)
    NUM_HEADS=4
    # Bidirectional= random.choice([True,False])
    # fc= random.choice([[64],[128],[256],[512],[1024],[2048],[4096]])

    LR= 0.001



    model = TransformerEncoderModel(shared_pre.vocab_size,embedding_dim=EMBED_DIM,num_layers=LSTM_LAYER,num_heads=NUM_HEADS,hidden_dim=HIDDEN_SIZE,dropout=0.1,output_dim=1
                                )
    model.to(device)


    # model.load_state_dict(torch.load('model/saved_models/lior_8_7_25.pt'))
    criterion = TripletLoss(alpha=1.0,beta=0.0,gamma=0)
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',patience=10)
    description = f'Trasformer encoder epochs:{EPOCHS} hidden: {HIDDEN_SIZE}, embed_dim: {EMBED_DIM} Layers: {LSTM_LAYER} heads {NUM_HEADS} '
    print(device)
    print(description)

    losses = []
    val_loss = []
    val_pearson= []
    val_spearman=[]

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

            outputs = model(rna_batch, rbp_batch, rna_mask, rbp_mask)
            # targets = scores_batch.view(-1, 1).to(device)

            loss = criterion(outputs, targets)
            if targets.shape != outputs.shape:
                print(f"score shape: {scores_batch.shape}, output shape: {targets.shape}")

            loss.backward()
            optimizer.step()

            curr_losses.append(loss.item())

        losses.append(np.mean(curr_losses))

        model.eval()
        with torch.no_grad():
            curr_losses = []
            y_orig = []
            y_hats = []

            for i, (( rna_batch, rbp_batch, rna_mask, rbp_mask), scores_batch) in tqdm(enumerate(val_loader)):
                rna_batch = rna_batch.to(device)
                rbp_batch = rbp_batch.to(device)
                rna_mask = rna_mask.to(device)
                rbp_mask = rbp_mask.to(device)
                targets = scores_batch.view(-1).to(device)  # חשוב להוריד מימד מיותר

                outputs = model(rna_batch, rbp_batch, rna_mask, rbp_mask)
                # targets = scores_batch.view(-1, 1).to(device)


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
    print(description)
    metrics(description,losses,val_loss,val_pearson,val_spearman)