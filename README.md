# Estimating Quantum Circuit Noise from Gate Counts

Course project, **Quantum Computing (25AM3504)**, School of Engineering, Dayananda Sagar University.

Chethan K S 


## What this project does

Quantum computers are noisy, so a circuit's real output differs from its ideal output. Running a circuit just to measure
that noise is slow and costly. This project trains machine-learning models to **predict a circuit's noise from the
gates it uses**, without running it.

- Data: the public **Qserv dataset** (35,000 random circuits, 4-10 qubits, simulated ideal vs noisy output for IBM Cairo and IBM Hanoi).
  The noise value is the cosine distance between the ideal and noisy outputs.
- Base paper: *Machine Learning Methods as Robust Quantum Noise Estimators* (arXiv:2409.14831). We reproduce it and extend it.
- Models: linear regression, histogram gradient boosting, XGBoost, a dense neural network, and a CNN with learned gate embeddings.
- Extras: extra inputs (qubit count, depth, derived counts), hyperparameter tuning, per-model plots, feature importance
  (coefficients, permutation, SHAP), and tests on another backend and on larger circuits.

## Main results (average error, MAE; lower is better; 80% of circuits used for training; input set D)

| Model | Cairo | Hanoi |
|---|---|---|
| Linear regression | 0.0780 | 0.0756 |
| Histogram gradient boosting | 0.0693 | 0.0286 |
| XGBoost | 0.0710 | 0.0298 |
| Dense neural network | 0.0742 | 0.0370 |
| CNN with embeddings | 0.0707 | 0.0300 |

Models do not transfer to the other backend (error about 0.19) or to larger circuits (0.14 on Cairo, 0.30 on Hanoi).
Noise is simulated and only two backends are used; no real quantum hardware was run.

## Repository layout

```
day1_experiments.py      first experiments: feature sets A-D, three models, 1% and 80% training
day2_experiments.py      5-split averages, plots, SHAP, cross-backend and unseen-size tests
day3_dense_nn.py         dense neural network
day3_cnn.py              CNN with learned gate embeddings (needs PyTorch)
day3_tuning.py           hyperparameter tuning for all model families
day3_model_figures.py    per-model predicted-vs-actual, residual and importance figures
demo_predict.py          short live demo: predict noise for unseen circuits
requirements.txt
results/                 result CSV files
figures/                 result figures
paper/                   IEEE-style paper (Word)
slides/                  presentation (PowerPoint)
```

## Setup

1. Install Python 3.10 or newer.
2. Install the libraries: `pip install -r requirements.txt`
3. Download the dataset from https://github.com/Jongarde/Qserv_dataset ("Code" > "Download ZIP") and extract it
   into this folder, so that a folder `Qserv_dataset-main` exists next to the scripts. It must contain
   `circuits_complete.csv` and the folders `cairo` and `hanoi`. (The dataset is not stored in this repository.)

## How to run (Windows PowerShell)

Run every command from inside the dataset folder, because the scripts read `circuits_complete.csv` and the
`cairo` / `hanoi` folders from the current folder:

```powershell
cd Qserv_dataset-main

# 1. quick live demo (a few seconds)
python ..\demo_predict.py circuits_complete.csv --backend Hanoi --n 8

# 2. main experiments
python ..\day1_experiments.py circuits_complete.csv
python ..\day2_experiments.py circuits_complete.csv
python ..\day3_dense_nn.py
python ..\day3_model_figures.py circuits_complete.csv      # figures go to figures_models\

# 3. CNN (quick test, then the full run)
python ..\day3_cnn.py --epochs 3
python ..\day3_cnn.py --epochs 40 --seeds 0 1 2

# 4. hyperparameter tuning (slow; the second line is a quick version)
python ..\day3_tuning.py circuits_complete.csv
python ..\day3_tuning.py circuits_complete.csv 4 3000
```

Each script writes its results (CSV files and images) into the folder it is run from. On macOS or Linux, use `/`
instead of `\`.

## Credits

Dataset and base paper: Gardeazabal-Gutierrez et al., *Machine Learning Methods as Robust Quantum Noise Estimators*,
arXiv:2409.14831, and the Qserv dataset (https://github.com/Jongarde/Qserv_dataset). Libraries: scikit-learn, XGBoost,
SHAP, PyTorch, matplotlib.
