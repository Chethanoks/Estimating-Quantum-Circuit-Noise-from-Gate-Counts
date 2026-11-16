"""
Per-model figures: predicted vs actual and feature importance for every model.

Run (from the folder with circuits_complete.csv):
    python ..\\day3_model_figures.py circuits_complete.csv
Needs: pip install pandas numpy scikit-learn xgboost matplotlib shap
Output (in ./figures_models/):
    fig4_pred_vs_actual_all_models.png   2 backends x 4 models
    fig5_importance_per_model.png        top features per tree/linear model, both backends
    fig6_residuals_all_models.png        error histograms per model
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "circuits_complete.csv"
OUT = "figures_models"
os.makedirs(OUT, exist_ok=True)
SEED = 42

df = pd.read_csv(CSV)
META = ["circuit", "Error_Cairo", "Error_Hanoi", "num_qubits", "depth"]
G = [c for c in df.columns if c not in META]
two = [g for g in ["cx", "cz", "cy", "ch", "cs", "csdg", "csx", "cp", "crx", "cry", "crz", "cu", "cu1", "cu3",
                   "swap", "iswap", "dcx", "ecr", "rxx", "ryy", "rzz", "rzx", "xx_minus_yy", "xx_plus_yy"] if g in df]
three = [g for g in ["ccx", "ccz", "cswap", "rccx", "c3sx", "rcccx"] if g in df]
non = [g for g in ["measure", "reset", "barrier"] if g in df]
one = [g for g in G if g not in two + three + non]
df["total_gates"] = df[one + two + three].sum(axis=1)
df["n_1q"] = df[one].sum(axis=1)
df["n_2q"] = df[two].sum(axis=1)
df["n_3q_plus"] = df[three].sum(axis=1)
df["ratio_2q"] = (df.n_2q + df.n_3q_plus) / df.total_gates.clip(lower=1)
df["gates_per_layer"] = df.total_gates / df.depth.clip(lower=1)
df["gates_per_qubit"] = df.total_gates / df.num_qubits
COLS = G + ["num_qubits", "depth", "total_gates", "n_1q", "n_2q", "n_3q_plus",
            "ratio_2q", "gates_per_layer", "gates_per_qubit"]   # feature set D
X = df[COLS]

MODELS = {
    "Linear regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(random_state=SEED),
    "XGBoost": lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
    "Dense NN": lambda: make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(80, 40), max_iter=300, early_stopping=True, random_state=SEED)),
}
BACKENDS = {"Cairo": "Error_Cairo", "Hanoi": "Error_Hanoi"}

fitted, preds, split = {}, {}, {}
for b, tgt in BACKENDS.items():
    y = df[tgt].values
    tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=SEED)
    split[b] = (tr, te)
    for name, make in MODELS.items():
        m = make().fit(X.iloc[tr], y[tr])
        fitted[(b, name)] = m
        preds[(b, name)] = m.predict(X.iloc[te])
        print(f"{b:6s} {name:22s} MAE {mean_absolute_error(y[te], preds[(b, name)]):.4f}", flush=True)

# ---- Figure 4: predicted vs actual, every model
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for i, b in enumerate(BACKENDS):
    y = df[BACKENDS[b]].values
    te = split[b][1]
    for j, name in enumerate(MODELS):
        p = preds[(b, name)]
        ax = axes[i, j]
        ax.scatter(y[te], p, s=2, alpha=0.2)
        lim = [min(y[te].min(), p.min()), max(y[te].max(), p.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(f"{b}: {name}\nMAE {mean_absolute_error(y[te], p):.4f}", fontsize=10)
        ax.set_xlabel("Actual error")
        if j == 0:
            ax.set_ylabel("Predicted error")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_pred_vs_actual_all_models.png", dpi=170)
plt.close()

# ---- Figure 6: residual histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
for i, b in enumerate(BACKENDS):
    y = df[BACKENDS[b]].values
    te = split[b][1]
    for j, name in enumerate(MODELS):
        r = preds[(b, name)] - y[te]
        ax = axes[i, j]
        ax.hist(r, bins=80, range=(-0.5, 0.5))
        ax.axvline(0, color="r", lw=1)
        ax.set_title(f"{b}: {name}", fontsize=10)
        ax.set_xlabel("Predicted - actual")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_residuals_all_models.png", dpi=170)
plt.close()

# ---- Figure 5: importance per model (linear coef, HistGB permutation, XGB SHAP)
def top(series, k=10):
    return series.sort_values(ascending=False).head(k)[::-1]

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for i, b in enumerate(BACKENDS):
    y = df[BACKENDS[b]].values
    tr, te = split[b]
    samp = np.random.RandomState(SEED).choice(te, size=2000, replace=False)
    # linear: absolute standardised coefficients
    lin = fitted[(b, "Linear regression")]
    coef = pd.Series(np.abs(lin[-1].coef_), index=COLS)
    t = top(coef)
    axes[i, 0].barh(t.index, t.values)
    axes[i, 0].set_title(f"{b}: Linear regression\n|standardised coefficient|", fontsize=10)
    # HistGB: permutation importance (MAE increase)
    hgb = fitted[(b, "HistGradientBoosting")]
    pi = permutation_importance(hgb, X.iloc[samp], y[samp], n_repeats=3, random_state=SEED,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    t = top(pd.Series(pi.importances_mean, index=COLS))
    axes[i, 1].barh(t.index, t.values)
    axes[i, 1].set_title(f"{b}: HistGradientBoosting\npermutation importance (MAE increase)", fontsize=10)
    # XGBoost: SHAP
    sv = shap.TreeExplainer(fitted[(b, "XGBoost")]).shap_values(X.iloc[samp])
    t = top(pd.Series(np.abs(sv).mean(axis=0), index=COLS))
    axes[i, 2].barh(t.index, t.values)
    axes[i, 2].set_title(f"{b}: XGBoost\nmean |SHAP value|", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_importance_per_model.png", dpi=170)
plt.close()
print(f"\nDone. Figures saved in ./{OUT}/")
