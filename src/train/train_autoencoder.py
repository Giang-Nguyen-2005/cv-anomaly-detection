from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this file directly: python src/train/train_autoencoder.py
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data.dataset import AnomalySequenceDataset
from src.models.autoencoder import ConvLSTMAutoEncoder


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(config: Dict[str, Any], dataset_path: str) -> DataLoader:
    data_cfg = config.get("data_loader", {})
    seq_length = int(data_cfg.get("sequence_length", 10))
    batch_size = int(data_cfg.get("batch_size", 8))
    resize_shape = tuple(data_cfg.get("resize_shape", [256, 256]))

    dataset = AnomalySequenceDataset(
        dataset_path=dataset_path,
        seq_length=seq_length,
        resize_shape=resize_shape,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def build_model(config: Dict[str, Any], device: torch.device) -> ConvLSTMAutoEncoder:
    model_cfg = config.get("model", {})
    model = ConvLSTMAutoEncoder(
        input_dim=int(model_cfg.get("input_dim", 1)),
        hidden_dims=model_cfg.get("hidden_dims", [64, 32, 32]),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
    )
    return model.to(device)


def train(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    dataloader = build_dataloader(config, args.dataset_path)
    print(f"Total sequences: {len(dataloader.dataset)}")
    if len(dataloader.dataset) == 0:
        raise ValueError("Dataset is empty. Please verify dataset_path and frame files.")

    model = build_model(config, device)
    hyper_cfg = config.get("hyperparameters", {})
    learning_rate = float(hyper_cfg.get("learning_rate", 1e-4))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in progress:
            batch = batch.to(device)  # (B, T, C, H, W)

            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / max(len(dataloader), 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch}: train_loss={epoch_loss:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "config": config,
            "dataset_path": args.dataset_path,
        }
        torch.save(checkpoint, save_dir / "last_autoencoder.pth")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, save_dir / "best_autoencoder.pth")
            print(f"Saved new best checkpoint (loss={best_loss:.6f})")

    with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM AutoEncoder for video anomaly detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/convlstm_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/raw/ucsd/UCSDped1/Train",
        help="Path containing sequence folders (each folder contains frames).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="weights/checkpoints",
        help="Directory for saving checkpoints.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
