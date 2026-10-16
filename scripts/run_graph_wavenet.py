"""Train Graph WaveNet-lite on METR-LA.

Logs to MLflow under urbanintel-week3.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from urbanintel.data.graph import load_adjacency
from urbanintel.data.metr_la import make_datasets
from urbanintel.evaluation.forecasting import masked_mae, masked_mape, masked_rmse
from urbanintel.models.graph_wavenet import GraphWaveNet

EXPERIMENT_NAME = "urbanintel-week3"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensors(ds):
    return TensorDataset(
        torch.from_numpy(ds.X).float(),
        torch.from_numpy(ds.y).float(),
        torch.from_numpy(ds.valid_mask).bool(),
    )


def masked_loss(pred, target, valid, eps: float = 1e-6):
    diff = (pred - target).abs() * valid.float()
    return diff.sum() / valid.float().sum().clamp(min=eps)


def evaluate(model, loader, adj, device):
    model.eval()
    ys, ps, vs = [], [], []
    with torch.no_grad():
        for X, y, valid in loader:
            X = X.to(device)
            adj = adj.to(device)
            pred = model(X, adj).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
            vs.append(valid.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(vs)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Device: {DEVICE}")
    adj_np = load_adjacency()
    adj = torch.from_numpy(adj_np).float()
    print(f"Adjacency: shape={adj.shape}, nonzero={int((adj > 0).sum())}")

    print("Loading METR-LA ...")
    datasets = make_datasets()
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    n_sensors = train_ds.X.shape[2]
    n_horizons = train_ds.y.shape[1]
    print(f"n_sensors={n_sensors}, n_horizons={n_horizons}")

    train_loader = DataLoader(to_tensors(train_ds), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(to_tensors(val_ds), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(to_tensors(test_ds), batch_size=BATCH_SIZE, shuffle=False)

    model = GraphWaveNet(
        n_sensors=n_sensors,
        n_horizons=n_horizons,
        input_length=train_ds.X.shape[1],
        hidden=64,
        n_blocks=2,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    with mlflow.start_run(run_name="graph-wavenet"):
        mlflow.log_params(
            {
                "model": "GraphWaveNet",
                "hidden": 64,
                "n_blocks": 2,
                "input_length": train_ds.X.shape[1],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "optimizer": "AdamW",
                "device": DEVICE,
                "n_params": n_params,
                "adaptive_adjacency": True,
                "fixed_adjacency": "METR-LA row-normalized adjacency (weight semantics not verified)",
            }
        )

        best_val_mae = float("inf")
        best_epoch = 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss, n_batches = 0.0, 0
            for X, y, valid in train_loader:
                X, y, valid = X.to(DEVICE), y.to(DEVICE), valid.to(DEVICE)
                adj_dev = adj.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X, adj_dev)
                loss = masked_loss(pred, y, valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss /= n_batches

            y_t, y_p, v = evaluate(model, val_loader, adj, DEVICE)
            val_mae = masked_mae(y_t, y_p, v)
            val_h = {}
            for h_idx, h in enumerate((3, 6, 12)):
                yt = y_t[:, h_idx, :][:, None, :]
                yp = y_p[:, h_idx, :][:, None, :]
                vv = v[:, h_idx, :][:, None, :]
                val_h[h] = masked_mae(yt, yp, vv)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_mae": val_mae,
                    **{f"val_mae_h{h}": val_h[h] for h in (3, 6, 12)},
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  "
                f"val_mae={val_mae:.4f}  (h3={val_h[3]:.3f}  "
                f"h6={val_h[6]:.3f}  h12={val_h[12]:.3f})"
            )
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_mae", best_val_mae)
        print(f"Best epoch: {best_epoch}, best val MAE: {best_val_mae:.4f}")

        model.load_state_dict(best_state)
        y_t, y_p, v = evaluate(model, test_loader, adj, DEVICE)

        for h_idx, h in enumerate((3, 6, 12)):
            yt = y_t[:, h_idx, :][:, None, :]
            yp = y_p[:, h_idx, :][:, None, :]
            vv = v[:, h_idx, :][:, None, :]
            mae = masked_mae(yt, yp, vv)
            rmse = masked_rmse(yt, yp, vv)
            mape = masked_mape(yt, yp, vv)
            mlflow.log_metrics(
                {f"gwnet_mae_h{h}": mae, f"gwnet_rmse_h{h}": rmse, f"gwnet_mape_h{h}": mape}
            )
            print(f"[horizon={h}] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        print("\n=== Model comparison (test MAE) ===")
        print(f"{'horizon':>8}  {'GWN':>8}  {'LightGBM':>10}  {'TCN':>8}  {'Persistence':>12}")
        # Reference MAEs from earlier runs, all computed on the same test set
        # (same chronological split, same target validity mask, same metric
        # implementations in urbanintel.evaluation.forecasting).
        # See MLflow experiments: urbanintel-week2 (LightGBM, TCN),
        # urbanintel-week2 (persistence).
        lgbm = {3: 3.1095, 6: 3.6992, 12: 4.3356}
        tcn = {3: 3.9721, 6: 4.1661, 12: 4.5935}
        per = {3: 4.0207, 6: 5.1001, 12: 6.8031}
        for h_idx, h in enumerate((3, 6, 12)):
            yt = y_t[:, h_idx, :][:, None, :]
            yp = y_p[:, h_idx, :][:, None, :]
            vv = v[:, h_idx, :][:, None, :]
            gwn = masked_mae(yt, yp, vv)
            print(f"{h:>8}  {gwn:>8.4f}  {lgbm[h]:>10.4f}  {tcn[h]:>8.4f}  {per[h]:>12.4f}")


if __name__ == "__main__":
    main()
