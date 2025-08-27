# Predicting RNA–RBP2 Scores

**Authors:** Nadav Weiss, Lior Alafi  
**Project type:** Deep Learning • Bioinformatics

This project predicts **binding scores** between RNA sequences and RBP2 proteins. The main entry point is `main.py`, which generates per‑RBP2 prediction files for a given set of RNAs.

---

## Predication Quickstart

```bash
python .\main.py <outfile> <rbps2.txt> <rna.txt>
```

* **`<outfile>`**: save results in this file **it's your responsibility to name it in different names`** 
* **`<rbps2.txt>`**: path to the RBP2 file; if your file uses the extension `.xt`, pass that exact filename.
* **`<rna.txt>`**: path to the RNA file.

**Notes**


* `<outfile>` corresponds to a single RBP2 and contains **one column of scores**, one per RNA (aligned to `<rna.txt>` line order).

---

**Example (Windows PowerShell):**

```powershell
mkdir output
python .\main.py output/ data\rbps2.txt data\rna.txt
```

**Example (macOS/Linux):**

```bash
mkdir -p output
python ./main.py output/ data/rbps2.txt data/rna.txt
```

---

## Detailed Setup

### 1) Prerequisites

* **Python** ≥ 3.9 (3.10+ recommended)
* **pip** (comes with Python ≥3.4)
* (Optional) **CUDA‑enabled GPU + PyTorch** for faster inference

### 2) Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If you don’t have `requirements.txt` yet, create one in your repo root and list your runtime libs (e.g., torch, numpy, pandas, scipy, tqdm, joblib, etc.).

---

## Troubleshooting

* **`Activate.ps1 is not digitally signed` (Windows PowerShell):**
  Run PowerShell **as Administrator** and execute:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
* **`pip: command not found`**: Ensure Python is on your PATH; try `python -m pip ...`.
* **CUDA not available**: PyTorch will run on CPU; ensure you installed the correct CUDA/PyTorch build if you want GPU.

## Model 
LSTM V2
epochs: 10
hidden dim(LSTM): 128
embedding dim:64
lstm Layers: 2
bidirectional: True 
fc: [64]
loss: MSE
over val 15% of training data: pearson_0.26414856735218467

# Project Structure (suggested)
```
project-root/
├─ README.md
├─ main.py
├─ requirements.txt
├─ training/
│  └─ lstm_training.py
├─ model/
│  ├─saved
│  │  ├─ amino_preproccessor.pt
│  │  ├─ amino_preproccessor.pt
│  │  ├─ rna_preproccessor.pt
│  │  ├─ LSTM_V2.pt
│  │  ├─ scaler.joblib
│  └─ lstm_model.py
├─ utils/
│  ├─ dataloader.py
│  └─ metrics.py
└─ output/                # <-- generated, must end with '/'
```

---

## Acknowledgements

This repository is part of a Deep Learning Bioinformatics project by **Nadav Weiss** and **Lior Alafi**, focusing on RNA–RBP2 interaction scoring.
