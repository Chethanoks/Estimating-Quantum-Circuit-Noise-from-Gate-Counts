"""
Hyperparameter tuning for ALL tabular models (feature set D, 80% train / 20% test, seed 42).

Run:   python day3_tuning.py circuits_complete.csv          (full: 10 settings, 8000 rows)
       python day3_tuning.py circuits_complete.csv 4 3000   (quick test)
Needs: pip install pandas numpy scikit-learn xgboost
Output: results_tuning.csv  (default MAE vs tuned MAE + best parameters, per backend and model)

Notes
- Plain LinearRegression has no hyperparameters, so we tune Ridge (alpha) as its tunable version.
- To keep run time short, the random search runs on an 8,000-row subsample of the training set
  (10 random settings, 3-fold CV). The best setting is then refit on the FULL 80% training set
  and scored once on the untouched 20% test set.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "circuits_complete.csv"
SEED = 42
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 10          # random settings tried per model
SEARCH_ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000  # rows used during the search
CV = 3

df = pd.read_csv(CSV)
META = ["circuit", "Error_Cairo", "Error_Hanoi", "num_qubits", "depth"]
GATES = [c for c in df.columns if c not in META]
TWO_Q = [g for g in ["cx", "cz", "cy", "ch", "cs", "csdg", "csx", "cp", "crx", "cry", "crz",
                     "cu", "cu1", "cu3", "swap", "iswap", "dcx", "ecr", "rxx", "ryy", "rzz",
                     "rzx", "xx_minus_yy", "xx_plus_yy"] if g in df.columns]
THREE_Q = [g for g in ["ccx", "ccz", "cswap", "rccx", "c3sx", "rcccx"] if g in df.columns]
NON_U = [g for g in ["measure", "reset", "barrier"] if g in df.columns]
ONE_Q = [g for g in GATES if g not in TWO_Q + THREE_Q + NON_U]
df["total_gates"] = df[ONE_Q + TWO_Q + THREE_Q].sum(axis=1)
df["n_1q"] = df[ONE_Q].sum(axis=1)
df["n_2q"] = df[TWO_Q].sum(axis=1)
df["n_3q_plus"] = df[THREE_Q].sum(axis=1)
df["ratio_2q"] = (df["n_2q"] + df["n_3q_plus"]) / df["total_gates"].clip(lower=1)
df["gates_per_layer"] = df["total_gates"] / df["depth"].clip(lower=1)
df["gates_per_qubit"] = df["total_gates"] / df["num_qubits"]
COLS = GATES + ["num_qubits", "depth", "total_gates", "n_1q", "n_2q", "n_3q_plus",
                "ratio_2q", "gates_per_layer", "gates_per_qubit"]
X = df[COLS].values

MODELS = {
    "Ridge (tuned LinearRegression)": dict(
        default=lambda: LinearRegression(),
        make=lambda: Ridge(),
        space={"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
    "HistGradientBoosting": dict(
        default=lambda: HistGradientBoostingRegressor(random_state=SEED),
        make=lambda: HistGradientBoostingRegressor(random_state=SEED),
        space={"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_iter": [200, 400, 800],
               "max_leaf_nodes": [15, 31, 63, 127], "min_samples_leaf": [10, 20, 50],
               "l2_regularization": [0, 0.1, 1.0]}),
    "XGBoost": dict(
        default=lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        make=lambda: XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        space={"n_estimators": [200, 400, 800], "max_depth": [4, 6, 8, 10],
               "learning_rate": [0.03, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0],
               "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5]}),
    "DenseNN (MLP)": dict(
        default=lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(80, 40), max_iter=300, early_stopping=True, random_state=SEED)),
        make=lambda: make_pipeline(StandardScaler(), MLPRegressor(
            max_iter=300, early_stopping=True, random_state=SEED)),
        space={"mlpregressor__hidden_layer_sizes": [(80, 40), (128, 64), (256, 128, 64)],
               "mlpregressor__alpha": [1e-5, 1e-4, 1e-3],
               "mlpregressor__learning_rate_init": [1e-3, 3e-3, 1e-2]}),
}

rows = []
for backend in ["Cairo", "Hanoi"]:
    y = df[f"Error_{backend}"].values
    tr, te = train_test_split(np.arange(len(df)), train_size=0.8, random_state=SEED)
    sub = np.random.RandomState(SEED).choice(tr, size=SEARCH_ROWS, replace=False)
    for name, cfg in MODELS.items():
        t0 = time.time()
        base = cfg["default"]().fit(X[tr], y[tr])
        p = base.predict(X[te])
        d_mae = mean_absolute_error(y[te], p)
        d_rmse = mean_squared_error(y[te], p) ** 0.5
        rs = RandomizedSearchCV(cfg["make"](), cfg["space"], n_iter=min(N_ITER, int(np.prod([len(v) for v in cfg["space"].values()]))),
                                cv=CV, scoring="neg_mean_absolute_error",
                                random_state=SEED, n_jobs=1)
        rs.fit(X[sub], y[sub])
        best = rs.best_estimator_.fit(X[tr], y[tr])          # refit on full training set
        p = best.predict(X[te])
        t_mae = mean_absolute_error(y[te], p)
        t_rmse = mean_squared_error(y[te], p) ** 0.5
        rows.append(dict(backend=backend, model=name,
                         default_MAE=round(d_mae, 4), tuned_MAE=round(t_mae, 4),
                         default_RMSE=round(d_rmse, 4), tuned_RMSE=round(t_rmse, 4),
                         best_params=str(rs.best_params_), seconds=round(time.time() - t0)))
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv("results_tuning.csv", index=False)
print("\nSaved results_tuning.csv")
print(out[["backend", "model", "default_MAE", "tuned_MAE"]].to_string(index=False))
