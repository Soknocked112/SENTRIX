#!/usr/bin/env python3
"""
02_train_tcn.py - Train Temporal Convolutional Network (TCN) on CICIDS 2017
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml

from data.dataset import get_dataloaders
from models.tcn import TCNThreatDetector


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train TCN Threat Detector on CICIDS 2017")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["processed_data_dir"]
    batch_size = args.batch_size or int(cfg["tcn"]["batch_size"])
    epochs = args.epochs or int(cfg["tcn"]["epochs"])
    lr = float(cfg["tcn"]["learning_rate"])
    weight_decay = float(cfg["tcn"]["weight_decay"])
    latent_dim = int(cfg["tcn"]["latent_dim"])
    num_classes = int(cfg["tcn"]["num_classes"])
    num_channels = cfg["tcn"]["num_channels"]
    kernel_size = int(cfg["tcn"]["kernel_size"])
    dropout = float(cfg["tcn"]["dropout"])
    device_str = cfg["tcn"].get("device", "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[*] Loading datasets from {data_dir}...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # Determine input feature count
    sample_x, _, _ = next(iter(train_loader))
    input_features = sample_x.shape[2]
    print(f"[+] Loaded data: input shape = {sample_x.shape} (Batch, Seq_Len, Features={input_features})")

    # Initialize TCN model
    print(f"[*] Initializing TCN Threat Detector (Channels={num_channels}, Latent={latent_dim}, Classes={num_classes})...")
    model = TCNThreatDetector(
        input_features=input_features,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        latent_dim=latent_dim,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"[*] Starting TCN training on {device} for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(ckpt_dir, "tcn_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "input_features": input_features,
                "num_channels": num_channels,
                "latent_dim": latent_dim,
                "num_classes": num_classes,
            }, best_path)
            print(f"  [+] Saved new best model checkpoint to {best_path} (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"[+] TCN Training finished in {total_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Evaluate on Test Set
    print("[*] Evaluating best model on unseen Test Set...")
    best_ckpt = torch.load(os.path.join(ckpt_dir, "tcn_best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[+] Final Test Set Evaluation -> Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")

    # Plot & Save Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("TCN Loss Convergence")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", color="royalblue")
    plt.plot(history["val_acc"], label="Val Accuracy", color="forestgreen")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TCN Classification Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = os.path.join(results_dir, "tcn_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
