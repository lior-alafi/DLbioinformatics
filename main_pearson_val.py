import datetime
import os
from pathlib import Path

import  torch

import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import pandas as pd
from model.load_model import load_model
from utils.dataloader4 import  CustomDataset, load_preprocessor
import argparse

from utils.metrics import pearson_evaluation

scaler = joblib.load("model/final/scaler.joblib")
BATCH_SIZE = 128

LR= 0.001



def main(withScore):

    scores = None
    if withScore:
        all_scores = np.loadtxt('data/train_scores.102578.txt', delimiter='\t').flatten().reshape(-1, 1)

    parser = argparse.ArgumentParser(
        description="Run inference and export rbpXX.txt grouped by amino_idx."
    )
    # parser.add_argument("out_dir", type=str, help="Output directory for CSV and rbpXX.txt files")
    # parser.add_argument("rbp_file", type=str, help="Path to RBP/amino sequences file")
    # parser.add_argument("rna_file", type=str, help="Path to RNA sequences file")
    # args = parser.parse_args()
    #
    # out_dir = Path(args.out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)

    rna_prep_emb_1 = load_preprocessor('model/final/rna_preproccessor.pt')
    amino_prep_emb_1 = load_preprocessor('model/final/amino_preproccessor.pt')

    # test_dataset = CustomDataset(args.rna_file,
    #                              args.rbp_file,
    #                              scores,
    #                              expected_score_dim=0,
    #                              rna_max_len=41,
    #                              amino_max_len=912,
    #                              score_transform=None,
    #                              preprocessing_rna=rna_prep_emb_1,
    #                              preprocessing_amino=amino_prep_emb_1)

    test_dataset = CustomDataset('data/validation_rna_seq.18100.txt',
                                 'data/validation_rbps2_seq.30.txt',
                                 'data/validation_scores.18100.txt', expected_score_dim=30,
                                 rna_max_len=41,
                                 amino_max_len=912,
                                 score_transform=scaler,
                                 preprocessing_rna=rna_prep_emb_1,
                                 preprocessing_amino=amino_prep_emb_1,

                                 )
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_rna = len(test_dataset.rna_sequences)
    n_rbp = len(test_dataset.amino_sequences)
    model_filepath = 'model/final/LSTM_V2_epochs_10_hidden_128_embed_dim_64_Layers_2_bidirectional_True_fc_[64]_MSE__pearson_0.26414856735218467_exp0_epoch_5.pt'
    model_filepath = 'model/final/LSTM_V2_epochs_10_hidden_128_embed_dim_8_Layers_2_bidirectional_True_fc_[64]_MSE__pearson_0.30778221790973176_exp0_epoch_0.pt'
    model_filepath = 'model/final/LSTM_V2_epochs_10_hidden_128_embed_dim_64_Layers_2_bidirectional_True_fc_[64]_MSE__pearson_0.2960118513175182_exp0_epoch_1.pt'
    model = load_model(model_filepath,rna_prep_emb_1,amino_prep_emb_1,{'EMBED_DIM': 64,
                                                                       'HIDDEN_SIZE': 128,
                                                                       'LSTM_LAYER': 2,
                                                                       'Bidirectional': True,
                                                                       'fc': [64]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for i, ((rna_batch, amino_batch, rna_mask, rbp_mask), scores_batch) in tqdm(enumerate(val_loader)):
            rna_batch = rna_batch.to(device)
            amino_batch = amino_batch.to(device)
            rna_mask = rna_mask.to(device)
            rbp_mask = rbp_mask.to(device)
            outputs = model(rna_batch, amino_batch, rna_mask, rbp_mask)

            y_pred_orig = torch.from_numpy(
                scaler.inverse_transform(outputs.detach().cpu().numpy())
            ).to(outputs.device, dtype=outputs.dtype)
            y_pred_orig = y_pred_orig.squeeze(-1)
            start = i * BATCH_SIZE
            gidx  = np.arange(start, start + len(y_pred_orig))                       # (B,)
            rna_idx   = (gidx // n_rbp).astype(int)                             # row-major
            amino_idx = (gidx %  n_rbp).astype(int)
            if  withScore:
                targets = scores_batch.view(-1, 1).to(device)
                targets = torch.from_numpy(
                    scaler.inverse_transform(targets.detach().cpu().numpy())
                ).to(targets.device, dtype=targets.dtype)
                targets = targets.squeeze(-1)
                rows.extend(zip(amino_idx.tolist(), rna_idx.tolist(), y_pred_orig.tolist(),targets.tolist()))
            else:
                rows.extend(zip(amino_idx.tolist(), rna_idx.tolist(), y_pred_orig.tolist()))

    # --- make a table and save ---
    df = None
    if withScore:
        df = pd.DataFrame(rows, columns=["amino_idx", "rna_idx", "score",'true_score'])
    else:
        df = pd.DataFrame(rows, columns=["amino_idx", "rna_idx", "score"])
    df = df.sort_values(["amino_idx", "rna_idx"])
    out_dir = "model/final/2.64/rbp_val/"
    os.makedirs(out_dir, exist_ok=True)
    pad = max(2, len(str(int(df["amino_idx"].max()))))
    pearsons=[]
    spearmans =[]
    for amino, g in df.groupby("amino_idx", sort=False):
        # Order by rna_idx and take scores as 1D array
        scores = g.sort_values("rna_idx")["score"].to_numpy()
        true_scores = g.sort_values("rna_idx")["true_score"].to_numpy()
        p_val = pearsonr(true_scores, scores)[0]
        s_val = spearmanr(true_scores, scores)[0]
        pearsons.append(p_val)
        spearmans.append(s_val)
        # If you want 0-based names: rbp00, rbp01, ...
        fname = os.path.join(out_dir, f"RBP2{int(amino+1):0{pad}d}.txt")

        np.savetxt(fname, scores, fmt="%.6f")    # one score per line
        print("wrote", fname, scores.shape)
        print(f'amino#{amino} pearson: {p_val} spearman: {s_val}')
    print(f'pearson {np.mean(pearsons)}')
    print(f'spearman {np.mean(spearmans)}')
    print(f'pearson nammean {np.nanmean(pearsons)}')
    print(f'spearman nanmean {np.nanmean(spearmans)}')


if __name__ == "__main__":
    start = datetime.datetime.now()
    withScore = True #False

    main(withScore)
    print(f'total time: {(datetime.datetime.now()-start)}')