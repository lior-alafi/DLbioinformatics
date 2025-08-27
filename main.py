from pathlib import Path
import argparse
import datetime
import os

from torch.utils.data import DataLoader
import joblib
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

from model.lstm_model import RecommendationModelV2
from utils.dataloader import CustomDataset, BasePreprocessing

BATCH_SIZE = 1024


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference and export rbpXX.txt grouped by amino_idx."
    )
    parser.add_argument("out_file", type=str, help="Output file for binding scores")
    parser.add_argument("rbp_file", type=str, help="Path to RBP/amino sequences file")
    parser.add_argument("rna_file", type=str, help="Path to RNA sequences file")
    args = parser.parse_args()
    return args


def reverse_scaler(outputs, scaler):
    y_pred_orig = torch.from_numpy(
        scaler.inverse_transform(outputs.detach().cpu().numpy())
    ).to(outputs.device, dtype=outputs.dtype)
    y_pred_orig = y_pred_orig.squeeze(-1)
    return y_pred_orig


def predict_binding_strength(device, model, scaler, test_dataset, val_loader):
    model.eval()
    binding_strengths = []
    with torch.no_grad():
        for i, ((rna_batch, amino_batch, rna_mask, rbp_mask), scores_batch) in enumerate(tqdm(val_loader)):
            rna_batch = rna_batch.to(device)
            amino_batch = amino_batch.to(device)
            rna_mask = rna_mask.to(device)
            rbp_mask = rbp_mask.to(device)

            outputs = model(rna_batch, amino_batch, rna_mask, rbp_mask)
            y_pred_orig = reverse_scaler(outputs, scaler)

            binding_strengths.extend(y_pred_orig.tolist())

    return binding_strengths


def write_results(binding_strengths, out_file):
    np.savetxt(out_file, binding_strengths, fmt="%.6f")  # one score per line
    print(f"Wrote {out_file} with {len(binding_strengths)} rows")


def main():
    arguments = parse_arguments()

    out_file = Path(arguments.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device} for prediction.')

    # Load embedding and standard scaler
    scaler = joblib.load("model/saved/scaler.joblib")
    rna_prep_emb_1 = BasePreprocessing.load_preprocessor('model/saved/rna_preproccessor.pt')
    amino_prep_emb_1 = BasePreprocessing.load_preprocessor('model/saved/amino_preproccessor.pt')
    # Load model
    model_filepath = 'model/saved/LSTM_V2.pt'
    model = RecommendationModelV2.load_model(
        model_filepath,
        rna_prep_emb_1,
        amino_prep_emb_1,
        embedding_dim=64,
        hidden_size=128,
        lstm_layers=2,
        fc_hidden_size=64,
        bidirectional=True,
    ).to(device)

    # Load test dataset
    test_dataset = CustomDataset(
        arguments.rna_file,
        arguments.rbp_file,
        None,
        expected_score_dim=0,
        rna_max_len=41,
        amino_max_len=912,
        score_transform=None,
        preprocessing_rna=rna_prep_emb_1,
        preprocessing_amino=amino_prep_emb_1
    )
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    binding_strengths = predict_binding_strength(device, model, scaler, test_dataset, val_loader)

    write_results(binding_strengths, out_file)


if __name__ == "__main__":
    start = datetime.datetime.now()
    try:
        main()
    finally:
        print(f'total time: {(datetime.datetime.now() - start)}')
