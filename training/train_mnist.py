"""
training/train_mnist.py
------------------------
PyTorch MNIST training script with:
  - Configurable epochs and batch size
  - Checkpoint save after every epoch  (./checkpoints/<job_name>/epoch_<N>.pt)
  - Resume training from a checkpoint (--resume-epoch)
  - Reports completion to the orchestrator via HTTP

Usage (standalone):
    python training/train_mnist.py \
        --job-id 1 --job-name my-run \
        --epochs 5 --batch-size 64 \
        --orchestrator-url http://127.0.0.1:8000 \
        --node-id node-1

Usage (resume from epoch 2):
    python training/train_mnist.py \
        --job-id 1 --job-name my-run \
        --epochs 5 --batch-size 64 \
        --resume-epoch 2 \
        --orchestrator-url http://127.0.0.1:8000 \
        --node-id node-1
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model definition ──────────────────────────────────────────────────────────

class MNISTNet(nn.Module):
    """Lightweight CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_data_loaders(batch_size: int, data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256,         shuffle=False, num_workers=0)
    return train_loader, test_loader


def save_checkpoint(model, optimizer, epoch: int, ckpt_dir: Path, job_name: str) -> str:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{job_name}_epoch_{epoch}.pt"
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logger.info("Checkpoint saved: %s", path)
    return str(path)


def load_checkpoint(path: str, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    logger.info("Resumed from checkpoint %s (epoch %d)", path, epoch)
    return epoch


def notify_orchestrator(orchestrator_url: str, job_id: int, node_id: str,
                        success: bool, checkpoint_path: str = None,
                        error: str = None, api_key: str = "") -> None:
    headers = {"X-API-Key": api_key} if api_key else {}
    # Try versioned endpoint first, fall back to legacy
    for path in ["/api/v1/job_complete", "/job_complete"]:
        url = f"{orchestrator_url}{path}"
        payload = {
            "job_id":          job_id,
            "node_id":         node_id,
            "success":         success,
            "checkpoint_path": checkpoint_path,
            "error":           error,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 404:
                continue
            logger.info("Notified orchestrator (%s): %s", path, resp.text[:120])
            return
        except Exception as exc:
            logger.warning("Failed to notify orchestrator at %s: %s", path, exc)
            return


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                "Epoch %d | batch %d/%d | loss=%.4f | acc=%.2f%%",
                epoch, batch_idx + 1, len(loader),
                total_loss / total, 100 * correct / total,
            )
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MNIST Trainer for GPU Agent")
    p.add_argument("--job-id",           type=int,   required=True)
    p.add_argument("--job-name",         type=str,   required=True)
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--resume-epoch",     type=int,   default=0,
                   help="Epoch to resume from (0 = fresh start)")
    p.add_argument("--checkpoint-path",  type=str,   default=None,
                   help="Path to existing checkpoint to resume from")
    p.add_argument("--checkpoint-dir",   type=str,   default="./checkpoints")
    p.add_argument("--data-dir",         type=str,   default="./data")
    p.add_argument("--orchestrator-url", type=str,   default="http://127.0.0.1:8000")
    p.add_argument("--node-id",          type=str,   required=True)
    p.add_argument("--api-key",          type=str,   default="",
                   help="API key for authenticating with the orchestrator")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    logger.info("Job %d | %s | epochs=%d | batch=%d | resume_epoch=%d",
                args.job_id, args.job_name, args.epochs,
                args.batch_size, args.resume_epoch)

    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir)

    # Model
    model     = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    last_ckpt   = None
    ckpt_dir    = Path(args.checkpoint_dir) / args.job_name

    # Resume from checkpoint if provided
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        resumed = load_checkpoint(args.checkpoint_path, model, optimizer)
        start_epoch = resumed + 1
        last_ckpt   = args.checkpoint_path
    elif args.resume_epoch > 0:
        # Try to find epoch checkpoint by convention
        candidate = ckpt_dir / f"{args.job_name}_epoch_{args.resume_epoch}.pt"
        if candidate.exists():
            load_checkpoint(str(candidate), model, optimizer)
            start_epoch = args.resume_epoch + 1
            last_ckpt   = str(candidate)

    t0 = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.2f%% "
                "val_loss=%.4f val_acc=%.2f%%",
                epoch, args.epochs,
                train_loss, 100 * train_acc,
                val_loss,   100 * val_acc,
            )
            last_ckpt = save_checkpoint(model, optimizer, epoch, ckpt_dir, args.job_name)

        total_time = time.time() - t0
        logger.info("Training complete in %.1fs", total_time)
        notify_orchestrator(
            args.orchestrator_url, args.job_id, args.node_id,
            success=True, checkpoint_path=last_ckpt,
            api_key=args.api_key,
        )
        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Training interrupted. Last checkpoint: %s", last_ckpt)
        notify_orchestrator(
            args.orchestrator_url, args.job_id, args.node_id,
            success=False, checkpoint_path=last_ckpt,
            error="Interrupted by user/system",
            api_key=args.api_key,
        )
        sys.exit(1)

    except Exception as exc:
        logger.exception("Training error: %s", exc)
        notify_orchestrator(
            args.orchestrator_url, args.job_id, args.node_id,
            success=False, checkpoint_path=last_ckpt,
            error=str(exc),
            api_key=args.api_key,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
