"""
CNN with learnable gate embeddings on the Qserv "image" data (the base paper's second pipeline).

Folder layout expected (same as the unzipped dataset):
    circuits_complete.csv
    hanoi/data(4q-hanoi).pt ... data(10q-hanoi).pt
    cairo/data(4q-cairo).pt  ... data(10q-cairo).pt

Run:   python day3_cnn.py                (from the dataset folder)
       python day3_cnn.py --epochs 5     (quick test first!)
Needs: pip install torch pandas numpy scikit-learn
Output: results_cnn.csv

What the data look like (checked on the files):
    X: (5000, 52, q)  float64 gate IDs 0..51  (52 layers x q qubits; 0 = empty cell)
    y: (5000,)        the error value (same numbers as Error_Hanoi / Error_Cairo in the CSV)
Circuits with fewer than 10 qubits are padded with ID 0 up to 10 qubits so all sizes can be
trained together.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--embed", type=int, default=8)
ap.add_argument("--csv", default="circuits_complete.csv")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
MAXQ, LAYERS, N_GATES = 10, 52, 52
df = pd.read_csv(args.csv)


def load_backend(backend):
    """Return X (N,52,10) long, y (N,), in the SAME circuit order as the CSV rows of that size."""
    Xs, ys = [], []
    for q in range(4, 11):
        # weights_only=False is needed on recent PyTorch because the file stores a TensorDataset
        data = torch.load(f"{backend}/data({q}q-{backend}).pt", weights_only=False)
        X, y = data.tensors
        X = X.long()
        y = y.float().view(-1)
        ref = df.loc[df.num_qubits == q, f"Error_{backend.capitalize()}"].values
        assert np.allclose(y.numpy(), ref, atol=1e-5), f"order mismatch for {q} qubits"
        pad = torch.zeros(X.shape[0], X.shape[1], MAXQ - q, dtype=torch.long)
        Xs.append(torch.cat([X, pad], dim=2))
        ys.append(y)
    return torch.cat(Xs), torch.cat(ys)


class CircuitCNN(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.emb = nn.Embedding(N_GATES, E)
        self.conv = nn.Sequential(
            nn.Conv2d(E, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        flat = 16 * (LAYERS // 4) * (MAXQ // 4)           # 16 * 13 * 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 80), nn.ReLU(),
                                  nn.Linear(80, 40), nn.ReLU(), nn.Linear(40, 1))

    def forward(self, x):                                  # x: (B, 52, 10) gate IDs
        z = self.emb(x).permute(0, 3, 1, 2)                # (B, E, 52, 10)  "image"
        return self.head(self.conv(z)).squeeze(1)


def run(X, y, train_frac, seed, epochs, E):
    torch.manual_seed(seed)
    tr, te = train_test_split(np.arange(len(y)), train_size=train_frac, random_state=seed)
    Xtr, ytr = X[tr].to(DEVICE), y[tr].to(DEVICE)
    Xte, yte = X[te].to(DEVICE), y[te].to(DEVICE)
    model = CircuitCNN(E).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 8), gamma=0.5)
    loss_fn = nn.MSELoss()
    bs = 256 if train_frac > 0.05 else 32
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=DEVICE)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1}/{epochs}", flush=True)
    fit_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(Xte[i:i + 2048]) for i in range(0, len(yte), 2048)])
    mae = (p - yte).abs().mean().item()
    rmse = ((p - yte) ** 2).mean().sqrt().item()
    return mae, rmse, fit_s


rows = []
for backend in ["cairo", "hanoi"]:
    X, y = load_backend(backend)
    print(f"{backend}: X {tuple(X.shape)}, y {tuple(y.shape)}")
    for frac in [0.01, 0.80]:
        res = [run(X, y, frac, s, args.epochs if frac > 0.05 else 100, args.embed)
               for s in args.seeds]
        mae, rmse, fit_s = np.mean(res, axis=0)
        rows.append(dict(backend=backend.capitalize(), train_frac=frac, model="CNN+embedding",
                         MAE=round(mae, 4), RMSE=round(rmse, 4), fit_s=round(fit_s, 1)))
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv("results_cnn.csv", index=False)
print("Saved results_cnn.csv")
