"""
Live demo: predict the noise of unseen circuits from their gate counts.

Run from the folder that contains circuits_complete.csv:
    python ..\\demo_predict.py circuits_complete.csv
    python ..\\demo_predict.py circuits_complete.csv --backend Cairo --n 10
    python ..\\demo_predict.py circuits_complete.csv --backend Hanoi --qubits 9 --n 8

It trains histogram gradient boosting on 80% of the circuits (about a few seconds),
then shows predicted vs actual noise (cosine distance) for random circuits from the
unseen 20%, and the overall average error (MAE) on that 20%.
Needs: pip install pandas numpy scikit-learn
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="circuits_complete.csv")
ap.add_argument("--backend", choices=["Cairo", "Hanoi"], default="Hanoi")
ap.add_argument("--n", type=int, default=8, help="how many example circuits to show")
ap.add_argument("--qubits", type=int, default=None, help="only show circuits with this many qubits (4-10)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(args.csv)
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
FEATURES = G + ["num_qubits", "depth", "total_gates", "n_1q", "n_2q", "n_3q_plus",
                "ratio_2q", "gates_per_layer", "gates_per_qubit"]   # feature set D

target = f"Error_{args.backend}"
tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=args.seed)
print(f"Backend: IBM {args.backend} | training on {len(tr):,} circuits, testing on {len(te):,} unseen circuits")
model = HistGradientBoostingRegressor(random_state=args.seed).fit(df.loc[tr, FEATURES], df.loc[tr, target])
pred_all = model.predict(df.loc[te, FEATURES])
mae = mean_absolute_error(df.loc[te, target], pred_all)
print(f"Average prediction error on the unseen circuits (MAE): {mae:.4f}\n")

test = df.loc[te].copy()
test["predicted"] = pred_all
test["abs_error"] = (test["predicted"] - test[target]).abs()
if args.qubits is not None:
    test = test[test.num_qubits == args.qubits]
    if test.empty:
        raise SystemExit(f"No test circuits with {args.qubits} qubits.")
show = test.sample(n=min(args.n, len(test)), random_state=args.seed)
out = show[["num_qubits", "depth", "total_gates", target, "predicted", "abs_error"]].rename(
    columns={"num_qubits": "qubits", "total_gates": "gates", target: "actual_noise",
             "predicted": "predicted_noise", "abs_error": "miss"})
print(out.round(3).to_string(index=False))
print("\nHow to read it: 'actual_noise' is the simulated noise (cosine distance between ideal and")
print("noisy output); 'predicted_noise' is the model's guess from gate counts only; 'miss' is the gap.")
