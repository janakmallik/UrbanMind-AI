"""Train a TCN forecaster on METR-LA.

Compares against persistence and LightGBM baselines.
Logs to MLflow under urbanintel-week2.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from urbanintel.data.metr_la import make_datasets
from urbanintel.evaluation.forecasting import masked_mae, masked_mape, masked_rmse
from urbanintel.models.tcn import TCNForecaster

EXPERIMENT_NAME = "urbanintel-week2"
SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensors(ds) -> TensorDataset:
    X = torch.from_numpy(ds.X).float()
    y = torch.from_numpy(ds.y).float()
    valid = torch.from_numpy(ds.valid_mask).bool()
    return TensorDataset(X, y, valid)


def masked_loss(pred, target, valid, eps: float = 1e-6):
    """Masked L1 loss: ignores positions where valid is False."""
    diff = (pred - target).abs() * valid.float()
    denom = valid.float().sum().clamp(min=eps)
    return diff.sum() / denom


def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_y = []
    all_valid = []
    with torch.no_grad():
        for X, y, valid in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            all_pred.append(pred)
            all_y.append(y.numpy())
            all_valid.append(valid.numpy())
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    valid = np.concatenate(all_valid, axis=0)
    return y_true, y_pred, valid


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Device: {DEVICE}")
    print("Loading METR-LA ...")
    datasets = make_datasets()
    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    n_sensors = train_ds.X.shape[2]
    n_horizons = train_ds.y.shape[1]
    print(f"n_sensors={n_sensors}, n_horizons={n_horizons}")
    print(f"Train: {train_ds.X.shape}, Val: {val_ds.X.shape}, Test: {test_ds.X.shape}")

    train_loader = DataLoader(to_tensors(train_ds), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(to_tensors(val_ds), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(to_tensors(test_ds), batch_size=BATCH_SIZE, shuffle=False)

    model = TCNForecaster(
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

    with mlflow.start_run(run_name="tcn"):
        mlflow.log_params(
            {
                "model": "TCN",
                "hidden": 64,
                "n_blocks": 4,
                "input_length": train_ds.X.shape[1],
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "optimizer": "AdamW",
                "device": DEVICE,
                "n_params": n_params,
            }
        )

        best_val_mae = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for X, y, valid in train_loader:
                X, y, valid = X.to(DEVICE), y.to(DEVICE), valid.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X)
                loss = masked_loss(pred, y, valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss /= n_batches

            y_true, y_pred, valid = evaluate(model, val_loader, DEVICE)
            val_mae = masked_mae(y_true, y_pred, valid)
            val_mae_h = {}
            for h_idx, h in enumerate((3, 6, 12)):
                y_t = y_true[:, h_idx, :][:, None, :]
                y_p = y_pred[:, h_idx, :][:, None, :]
                v = valid[:, h_idx, :][:, None, :]
                val_mae_h[h] = masked_mae(y_t, y_p, v)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_mae": val_mae,
                    "val_mae_h3": val_mae_h[3],
                    "val_mae_h6": val_mae_h[6],
                    "val_mae_h12": val_mae_h[12],
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  "
                f"val_mae={val_mae:.4f}  (h3={val_mae_h[3]:.3f}  "
                f"h6={val_mae_h[6]:.3f}  h12={val_mae_h[12]:.3f})"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best and evaluate on test
        model.load_state_dict(best_state)
        y_true, y_pred, valid = evaluate(model, test_loader, DEVICE)

        for h_idx, h in enumerate((3, 6, 12)):
            y_t = y_true[:, h_idx, :][:, None, :]
            y_p = y_pred[:, h_idx, :][:, None, :]
            v = valid[:, h_idx, :][:, None, :]
            mae = masked_mae(y_t, y_p, v)
            rmse = masked_rmse(y_t, y_p, v)
            mape = masked_mape(y_t, y_p, v)
            mlflow.log_metrics(
                {f"tcn_mae_h{h}": mae, f"tcn_rmse_h{h}": rmse, f"tcn_mape_h{h}": mape}
            )
            print(f"[horizon={h}] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        # Comparison — persistence computed on the SAME test targets
        print("\n=== TCN vs persistence (computed on same test targets) ===")
        print(f"{'horizon':>8}  {'TCN':>10}  {'Persistence':>12}")
        for h_idx, h in enumerate((3, 6, 12)):
            y_t = y_true[:, h_idx, :][:, None, :]
            y_p = y_pred[:, h_idx, :][:, None, :]
            y_per = test_ds.X[:, -1, :][:, None, :]  # persistence
            v = valid[:, h_idx, :][:, None, :]
            tcn_mae = masked_mae(y_t, y_p, v)
            per_mae = masked_mae(y_t, y_per, v)
            print(f"{h:>8}  {tcn_mae:>10.4f}  {per_mae:>12.4f}")

        # Save model
        torch.save(model.state_dict(), "tcn_best.pt")
        mlflow.log_artifact("tcn_best.pt")
        os.remove("tcn_best.pt")


if __name__ == "__main__":
    main()
