"""
Day 1: noise estimation on the public Qserv dataset (35,000 circuits).

Run:   python day1_experiments.py path/to/circuits_complete.csv
Needs: pip install pandas numpy scikit-learn xgboost matplotlib

Target: Error_Cairo / Error_Hanoi (cosine distance between ideal and noisy output).

Feature sets
  A  gate counts only                       (what the base paper does, with measure/reset)
  B  A without measure                      (paper's "gates removed" experiment)
  C  A + num_qubits + depth                 (your extra #1)
  D  C + derived features                   (totals, 1q/2q/3q+ counts, 2q ratio, gates per layer)
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "circuits_complete.csv"

df = pd.read_csv(CSV)

META = ["circuit", "Error_Cairo", "Error_Hanoi", "num_qubits", "depth"]
GATES = [c for c in df.columns if c not in META]

# Gate groups by how many qubits they act on (used for derived features)
TWO_Q = ["cx", "cz", "cy", "ch", "cs", "csdg", "csx", "cp", "crx", "cry", "crz",
         "cu", "cu1", "cu3", "swap", "iswap", "dcx", "ecr", "rxx", "ryy", "rzz",
         "rzx", "xx_minus_yy", "xx_plus_yy"]
THREE_PLUS_Q = ["ccx", "ccz", "cswap", "rccx", "c3sx", "rcccx"]
TWO_Q = [g for g in TWO_Q if g in df.columns]
THREE_PLUS_Q = [g for g in THREE_PLUS_Q if g in df.columns]
NON_UNITARY = [g for g in ["measure", "reset", "barrier"] if g in df.columns]
ONE_Q = [g for g in GATES if g not in TWO_Q + THREE_PLUS_Q + NON_UNITARY]

# Derived features
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
    "LinearRegression": lambda: LinearRegression(),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(random_state=SEED),
    "XGBoost": lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
}

TARGETS = {"Cairo": "Error_Cairo", "Hanoi": "Error_Hanoi"}
TRAIN_FRACTIONS = [0.01, 0.80]  # paper's two regimes

rows = []
for backend, target in TARGETS.items():
    y = df[target].values
    for frac in TRAIN_FRACTIONS:
        # same split for every feature set and model -> fair comparison
        idx_tr, idx_te = train_test_split(
            np.arange(len(df)), train_size=frac, random_state=SEED
        )
        for fs_name, cols in FEATURE_SETS.items():
            X = df[cols].values
            for m_name, make in MODELS.items():
                model = make()
                t0 = time.time()
                model.fit(X[idx_tr], y[idx_tr])
                t_fit = time.time() - t0
                t0 = time.time()
                pred = model.predict(X[idx_te])
                t_pred = time.time() - t0
                rows.append({
                    "backend": backend, "train_frac": frac,
                    "features": fs_name, "model": m_name,
                    "MAE": mean_absolute_error(y[idx_te], pred),
                    "RMSE": mean_squared_error(y[idx_te], pred) ** 0.5,
                    "fit_s": round(t_fit, 3), "pred_s": round(t_pred, 3),
                })
                print(f"{backend:6s} {frac:4.2f} {fs_name:20s} {m_name:22s} "
                      f"MAE={rows[-1]['MAE']:.4f} RMSE={rows[-1]['RMSE']:.4f}")

res = pd.DataFrame(rows)
res.to_csv("results_day1.csv", index=False)
print("\nSaved results_day1.csv")
print(res.pivot_table(index=["backend", "train_frac", "features"],
                      columns="model", values="MAE").round(4))
