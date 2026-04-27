#!/usr/bin/env python3
"""Train Ternary Neural Networks (TNNs) using Trained Ternary Quantization (TTQ).

Trains 45 networks: 3 architectures x 3 sparsity levels x 5 random seeds.
All networks are trained on MNIST with the TTQ algorithm.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path


# ─── Architecture definitions ───────────────────────────────────────────────

ARCHITECTURES = {
    "small":  [784, 64, 64, 64, 10],
    "medium": [784, 128, 256, 256, 128, 10],
    "large":  [784, 256, 512, 512, 512, 256, 10],
}

SPARSITY_LEVELS = [0.15, 0.30, 0.50]
SEEDS = [0, 1, 2, 3, 4]

# ─── TTQ hyperparameters ────────────────────────────────────────────────────

EPOCHS = 160
BATCH_SIZE = 128
LR_INITIAL = 0.01
LR_MOMENTUM = 0.9
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = [80, 120]


# ─── TTQ Module ─────────────────────────────────────────────────────────────

class TTQLinear(nn.Module):
    """Linear layer with Trained Ternary Quantization.

    Weights are quantized to {-W_n, 0, W_p} during forward pass.
    W_p and W_n are learnable per-layer scaling factors.
    """

    def __init__(self, in_features, out_features, target_sparsity=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = target_sparsity

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.W_p = nn.Parameter(torch.tensor(1.0))
        self.W_n = nn.Parameter(torch.tensor(1.0))

        nn.init.kaiming_uniform_(self.weight)

    def _compute_threshold(self):
        """Compute the threshold delta for ternary quantization."""
        abs_weight = self.weight.abs()
        # Use target_sparsity-th percentile as threshold
        threshold = torch.quantile(abs_weight, self.target_sparsity)
        return threshold

    def _quantize(self):
        """Quantize weights to ternary values {-W_n, 0, W_p}."""
        delta = self._compute_threshold()
        pos_mask = (self.weight > delta).float()
        neg_mask = (self.weight < -delta).float()
        ternary = self.W_p * pos_mask - self.W_n * neg_mask
        return ternary

    def forward(self, x):
        q_weight = self._quantize()
        return nn.functional.linear(x, q_weight, self.bias)


class TTQNetwork(nn.Module):
    """Fully connected TNN with TTQ quantization."""

    def __init__(self, layer_sizes, target_sparsity=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                TTQLinear(layer_sizes[i], layer_sizes[i + 1],
                          target_sparsity=target_sparsity)
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.sign(x)  # Binary activation (step function)
        return x

    def extract_ternary_weights(self):
        """Extract quantized ternary weights {-1, 0, +1} for verification."""
        weights = []
        for layer in self.layers:
            delta = layer._compute_threshold()
            w = layer.weight.data
            ternary = torch.zeros_like(w)
            ternary[w > delta] = 1.0
            ternary[w < -delta] = -1.0
            weights.append(ternary)
        return weights


# ─── Training ───────────────────────────────────────────────────────────────

def get_data_loaders():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarize inputs
    ])
    train_set = datasets.MNIST("./data", train=True, download=True,
                                transform=transform)
    test_set = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, test_loader, device):
    model.train(False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return 100.0 * correct / total


def train_network(arch_name, layer_sizes, sparsity, seed, output_dir, device):
    """Train a single TNN and save the ternary weights."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = TTQNetwork(layer_sizes, target_sparsity=sparsity).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR_INITIAL,
                          momentum=LR_MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)

    train_loader, test_loader = get_data_loaders()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 40 == 0 or epoch == EPOCHS:
            test_acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch}/{EPOCHS}: "
                  f"loss={train_loss:.4f} train_acc={train_acc:.1f}% "
                  f"test_acc={test_acc:.1f}%")

    test_acc = evaluate(model, test_loader, device)

    # Extract and save ternary weights
    ternary_weights = model.extract_ternary_weights()
    actual_sparsity = sum(
        (w == 0).float().mean().item() for w in ternary_weights
    ) / len(ternary_weights)

    name = f"{arch_name}_sp{int(sparsity*100)}_seed{seed}"
    save_dir = Path(output_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save weights as numpy arrays
    for i, w in enumerate(ternary_weights):
        torch.save(w, save_dir / f"layer_{i}_weights.pt")

    # Save biases
    for i, layer in enumerate(model.layers):
        torch.save(layer.bias.data, save_dir / f"layer_{i}_bias.pt")

    # Save metadata
    metadata = {
        "architecture": arch_name,
        "layer_sizes": layer_sizes,
        "sparsity_target": sparsity,
        "sparsity_actual": actual_sparsity,
        "seed": seed,
        "test_accuracy": test_acc,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_initial": LR_INITIAL,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {name} (test_acc={test_acc:.1f}%, "
          f"sparsity={actual_sparsity:.1%})")
    return metadata


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train TNN networks with TTQ")
    parser.add_argument("--output", type=str, default="benchmarks/networks/",
                        help="Output directory for trained networks")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["small", "medium", "large"],
                        help="Train only one architecture (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    architectures = {args.arch: ARCHITECTURES[args.arch]} if args.arch \
        else ARCHITECTURES

    all_metadata = []
    for arch_name, layer_sizes in architectures.items():
        for sparsity in SPARSITY_LEVELS:
            for seed in SEEDS:
                print(f"Training {arch_name} sparsity={sparsity:.0%} "
                      f"seed={seed}")
                meta = train_network(
                    arch_name, layer_sizes, sparsity, seed,
                    args.output, device)
                all_metadata.append(meta)

    # Save summary
    with open(Path(args.output) / "training_summary.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nTrained {len(all_metadata)} networks.")
    accs = [m["test_accuracy"] for m in all_metadata]
    print(f"Test accuracy: {min(accs):.1f}% - {max(accs):.1f}%")


if __name__ == "__main__":
    main()
