"""
Day 2: reliability, plots, feature importance, generalisation tests.

Run:   python day2_experiments.py circuits_complete.csv
Needs: pip install pandas numpy scikit-learn xgboost matplotlib
Optional: pip install shap   (if it fails, the script falls back to permutation importance)

Outputs (all saved in the current folder):
  results_day2_seeds.csv        mean/std MAE over 5 random splits
  fig1_mae_bars.png             MAE by feature set and model (error bars = std)
  fig2_pred_vs_actual.png       predicted vs actual error, best model
  fig3_feature_importance.png   top features (SHAP or permutation)
  results_day2_generalisation.csv  cross-backend + unseen-qubit-count tests
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

CSV = sys.argv[1] if len(sys.argv) > 1 else "circuits_complete.csv"
SEEDS = [0, 1, 2, 3, 4]
TARGETS = {"Cairo": "Error_Cairo", "Hanoi": "Error_Hanoi"}

# ---------------------------------------------------------------- features
df = pd.read_csv(CSV)
META = ["circuit", "Error_Cairo", "Error_Hanoi", "num_qubits", "depth"]
GATES = [c for c in df.columns if c not in META]

TWO_Q = ["cx", "cz", "cy", "ch", "cs", "csdg", "csx", "cp", "crx", "cry", "crz",
         "cu", "cu1", "cu3", "swap", "iswap", "dcx", "ecr", "rxx", "ryy", "rzz",
         "rzx", "xx_minus_yy", "xx_plus_yy"]
THREE_PLUS_Q = ["ccx", "ccz", "cswap", "rccx", "c3sx", "rcccx"]
TWO_Q = [g for g in TWO_Q if g in df.columns]
THREE_PLUS_Q = [g for g in THREE_PLUS_Q if g in df.columns]
NON_UNITARY = [g for g in ["measure", "reset", "barrier"] if g in df.columns]
ONE_Q = [g for g in GATES if g not in TWO_Q + THREE_PLUS_Q + NON_UNITARY]

df["total_gates"] = df[ONE_Q + TWO_Q + THREE_PLUS_Q].sum(axis=1)
df["n_1q"] = df[ONE_Q].sum(axis=1)
df["n_2q"] = df[TWO_Q].sum(axis=1)
df["n_3q_plus"] = df[THREE_PLUS_Q].sum(axis=1)
df["ratio_2q"] = (df["n_2q"] + df["n_3q_plus"]) / df["total_gates"].clip(lower=1)
df["gates_per_layer"] = df["total_gates"] / df["depth"].clip(lower=1)
df["gates_per_qubit"] = df["total_gates"] / df["num_qubits"]
DERIVED = ["total_gates", "n_1q", "n_2q", "n_3q_plus", "ratio_2q",
           "gates_per_layer", "gates_per_qubit"]

FEATURE_SETS = {
    "A_gate_counts": GATES,
    "B_no_measure": [g for g in GATES if g not in NON_UNITARY],
    "C_plus_qubits_depth": GATES + ["num_qubits", "depth"],
    "D_plus_derived": GATES + ["num_qubits", "depth"] + DERIVED,
}
MODELS = {
    "LinearRegression": lambda s: LinearRegression(),
    "HistGradientBoosting": lambda s: HistGradientBoostingRegressor(random_state=s),
    "XGBoost": lambda s: XGBRegressor(random_state=s, n_jobs=-1, verbosity=0),
}

def mae(y, p): return mean_absolute_error(y, p)
def rmse(y, p): return mean_squared_error(y, p) ** 0.5

# ------------------------------------------------- Part 1: multi-seed runs
print("Part 1/4: multi-seed runs (5 seeds) ...")
rows = []
for backend, target in TARGETS.items():
    y = df[target].values
    for frac in [0.01, 0.80]:
        for seed in SEEDS:
            tr, te = train_test_split(np.arange(len(df)), train_size=frac, random_state=seed)
            for fs, cols in FEATURE_SETS.items():
                X = df[cols].values
                for mn, make in MODELS.items():
                    m = make(seed).fit(X[tr], y[tr])
                    p = m.predict(X[te])
                    rows.append(dict(backend=backend, train_frac=frac, seed=seed,
                                     features=fs, model=mn, MAE=mae(y[te], p), RMSE=rmse(y[te], p)))
    print(f"  done {backend}")
seeds_df = pd.DataFrame(rows)
summary = (seeds_df.groupby(["backend", "train_frac", "features", "model"])
           .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"))
           .reset_index().round(4))
summary.to_csv("results_day2_seeds.csv", index=False)
print(summary[summary.train_frac == 0.80].to_string(index=False))

# ---------------------------------------------------- Part 2: figures 1+2
print("\nPart 2/4: plots ...")
s80 = summary[summary.train_frac == 0.80]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
width = 0.25
for ax, backend in zip(axes, TARGETS):
    sub = s80[s80.backend == backend]
    for i, mn in enumerate(MODELS):
        d = sub[sub.model == mn].set_index("features").loc[list(FEATURE_SETS)]
        ax.bar(np.arange(len(FEATURE_SETS)) + i * width, d.MAE_mean, width,
               yerr=d.MAE_std, capsize=3, label=mn)
    ax.set_xticks(np.arange(len(FEATURE_SETS)) + width)
    ax.set_xticklabels([f.split("_")[0] for f in FEATURE_SETS])
    ax.set_title(f"{backend} (80% train, mean ± std over {len(SEEDS)} seeds)")
    ax.set_xlabel("Feature set (A gates, B no measure, C +qubits/depth, D +derived)")
    ax.set_ylabel("MAE (lower is better)")
axes[0].legend()
plt.tight_layout(); plt.savefig("fig1_mae_bars.png", dpi=200); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (backend, target) in zip(axes, TARGETS.items()):
    y = df[target].values
    tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=42)
    X = df[FEATURE_SETS["D_plus_derived"]].values
    m = HistGradientBoostingRegressor(random_state=42).fit(X[tr], y[tr])
    p = m.predict(X[te])
    ax.scatter(y[te], p, s=3, alpha=0.25)
    lim = [min(y[te].min(), p.min()), max(y[te].max(), p.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_title(f"{backend}: HistGB, feature set D (MAE={mae(y[te], p):.4f})")
    ax.set_xlabel("Actual error"); ax.set_ylabel("Predicted error")
plt.tight_layout(); plt.savefig("fig2_pred_vs_actual.png", dpi=200); plt.close()

# ----------------------------------------- Part 3: feature importance
print("Part 3/4: feature importance ...")
cols = FEATURE_SETS["D_plus_derived"]
X = df[cols]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (backend, target) in zip(axes, TARGETS.items()):
    y = df[target].values
    tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=42)
    try:
        import shap
        m = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0).fit(X.iloc[tr], y[tr])
        samp = X.iloc[te].sample(2000, random_state=42)
        sv = shap.TreeExplainer(m).shap_values(samp)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols)
        label = "mean |SHAP value|"
    except Exception as e:
        print(f"  (SHAP unavailable for {backend}: {type(e).__name__}; using permutation importance)")
        m = HistGradientBoostingRegressor(random_state=42).fit(X.iloc[tr], y[tr])
        samp = X.iloc[te].sample(3000, random_state=42)
        r = permutation_importance(m, samp, y[samp.index], n_repeats=5,
                                   random_state=42, scoring="neg_mean_absolute_error")
        imp = pd.Series(r.importances_mean, index=cols)
        label = "permutation importance (MAE increase)"
    top = imp.sort_values(ascending=False).head(12)[::-1]
    ax.barh(top.index, top.values)
    ax.set_title(f"{backend}: top 12 features"); ax.set_xlabel(label)
    print(f"  {backend} top 5:", ", ".join(top.index[::-1][:5]))
plt.tight_layout(); plt.savefig("fig3_feature_importance.png", dpi=200); plt.close()

# ------------------------------------ Part 4: generalisation tests
print("\nPart 4/4: generalisation tests ...")
gen = []
X = df[FEATURE_SETS["D_plus_derived"]].values
print("  Correlation between Error_Cairo and Error_Hanoi: "
      f"{df['Error_Cairo'].corr(df['Error_Hanoi']):.3f}")

# (a) cross-backend: train on Cairo errors, test on Hanoi errors (same unseen circuits)
tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=42)
for mn, make in MODELS.items():
    for name, ytr, yte in [("Cairo->Cairo", "Error_Cairo", "Error_Cairo"),
                           ("Cairo->Hanoi", "Error_Cairo", "Error_Hanoi"),
                           ("Hanoi->Hanoi", "Error_Hanoi", "Error_Hanoi"),
                           ("Hanoi->Cairo", "Error_Hanoi", "Error_Cairo")]:
        m = make(42).fit(X[tr], df[ytr].values[tr])
        p = m.predict(X[te])
        gen.append(dict(test="cross_backend", setting=name, model=mn,
                        MAE=round(mae(df[yte].values[te], p), 4)))

# (b) unseen qubit counts: train on 4-8 qubits, test on 9-10 qubits
train_mask = (df["num_qubits"] <= 8).values
for backend, target in TARGETS.items():
    y = df[target].values
    for mn, make in MODELS.items():
        m = make(42).fit(X[train_mask], y[train_mask])
        p = m.predict(X[~train_mask])
        gen.append(dict(test="train_4-8q_test_9-10q", setting=backend, model=mn,
                        MAE=round(mae(y[~train_mask], p), 4)))
gen_df = pd.DataFrame(gen)
gen_df.to_csv("results_day2_generalisation.csv", index=False)
print(gen_df.pivot_table(index=["test", "setting"], columns="model", values="MAE").round(4))
print("\nDone. Saved: results_day2_seeds.csv, results_day2_generalisation.csv, "
      "fig1_mae_bars.png, fig2_pred_vs_actual.png, fig3_feature_importance.png")
