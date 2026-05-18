import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
TPV trajectory when training ResNet-18 on CIFAR-100
====================================================

Pipeline:
  1. Train a CIFAR-adapted ResNet-18 on CIFAR-100 with a standard high-accuracy
     SGD recipe (cosine schedule, weight decay, momentum, augmentation) with a 
     fraction of training set class labels corrupted randomly.
  2. Every TPV_EVERY epochs, snapshot the model and estimate empirical
     training-set TPV via noisy-logit fine-tuning on a fixed 10k subset
     (no augmentation; channel-wise normalization only).
  3. Plot training accuracy, validation accuracy, and training-set TPV
     against epoch on a dual-axis figure.

Usage:
    python tpv_trajectory_cifar100.py --labelsmooth 0.1 --noise_ratio 0.3
"""

import argparse
import copy
import pickle as pkl
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------------------
# Argument parsing
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="TPV trajectory on CIFAR-100 with ResNet-18."
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs.")
    parser.add_argument("--tpv_every", type=int, default=5,
                        help="Compute TPV every N epochs (also at epoch 0).")
    parser.add_argument("--tpv_subset_size", type=int, default=10000,
                        help="Number of training samples for TPV estimation.")
    parser.add_argument("--R", type=int, default=5,
                        help="Number of noisy fine-tuning runs per TPV estimate.")
    parser.add_argument("--max_epochs_noisy", type=int, default=3,
                        help="Fine-tuning epochs per noisy run.")
    parser.add_argument("--lr_noisy", type=float, default=1e-4,
                        help="Learning rate for noisy fine-tuning.")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="sigma multiplier for adaptive logit noise.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tpv_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--no_adaptive_noise_scaling", action="store_true")
    parser.add_argument("--noise_ratio", type=float, default=0.30,
                        help="Fraction of training labels to corrupt.")
    parser.add_argument("--labelsmooth", type=float, default=0.0,
                        help="Label smoothing for the main training run.")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


args = get_args()
ADAPTIVE_NOISE_SCALING = not args.no_adaptive_noise_scaling


# ----------------------------
# Reproducibility & device
# ----------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("./results/", exist_ok=True)
os.makedirs("./results/plots/", exist_ok=True)


# ----------------------------
# CIFAR-adapted ResNet-18
# ----------------------------
class _ResBasicBlock(nn.Module):
    """Basic residual block: two 3x3 convs + BN + ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class CIFARResNet18(nn.Module):
    """
    CIFAR-style ResNet-18: 3x3 stem (no max-pool), 4 stages of [2,2,2,2]
    BasicBlocks with widths [64, 128, 256, 512].
    """
    def __init__(self, num_classes=100):
        super().__init__()
        C = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(C,    C,    2, stride=1)
        self.stage2 = self._make_stage(C,    C*2,  2, stride=2)
        self.stage3 = self._make_stage(C*2,  C*4,  2, stride=2)
        self.stage4 = self._make_stage(C*4,  C*8,  2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(C * 8, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, stride):
        layers = [_ResBasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(_ResBasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------
# Data loading
# ----------------------------
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# TPV subset and validation: normalization only, no augmentation.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

_train_set_aug_base = torchvision.datasets.CIFAR100(
    "./data", train=True,  download=True, transform=train_transform)

# Randomly flip a configurable fraction of training labels to a different class.
class _NoisyLabelDataset(Dataset):
    def __init__(self, dataset, noise_ratio, num_classes, seed):
        self.dataset = dataset
        rng_noise = np.random.default_rng(seed)
        n = len(dataset)
        self.targets = list(dataset.targets)
        noisy_idx = rng_noise.choice(n, size=int(noise_ratio * n), replace=False)
        for i in noisy_idx:
            orig = self.targets[i]
            choices = [c for c in range(num_classes) if c != orig]
            self.targets[i] = int(rng_noise.choice(choices))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.targets[idx]

train_set_aug = _NoisyLabelDataset(
    _train_set_aug_base, noise_ratio=args.noise_ratio, num_classes=100,
    seed=args.seed)

train_set_clean = torchvision.datasets.CIFAR100(
    "./data", train=True,  download=True, transform=eval_transform)
val_set         = torchvision.datasets.CIFAR100(
    "./data", train=False, download=True, transform=eval_transform)

# Fixed 10k subset of the (un-augmented) training set for TPV estimation.
# TPV uses *clean* images (clean labels are not actually used by TPV, but the
# images come from the same indices regardless).
rng = np.random.default_rng(args.seed)
tpv_indices = rng.choice(len(train_set_clean), size=args.tpv_subset_size,
                         replace=False)
tpv_subset = Subset(train_set_clean, tpv_indices)

train_loader_aug = DataLoader(
    train_set_aug, batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=False)
val_loader = DataLoader(
    val_set, batch_size=args.tpv_batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

print(f"CIFAR-100: {len(train_set_aug)} train ({args.noise_ratio:.0%} label "
      f"noise), {len(val_set)} val, TPV subset size: {len(tpv_subset)}.")


# ----------------------------
# Standard accuracy / loss eval
# ----------------------------
@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


# ----------------------------
# TPV estimation (noisy-logit fine-tuning)
# ----------------------------
class _LogitNoiseDataset(Dataset):
    """
    Yields (image, noisy_logit, teacher_logit, label) for the TPV subset.
    Indexed identically to the underlying Subset.
    """
    def __init__(self, subset, teacher_logits, noisy_logits, labels):
        self.subset = subset
        self.teacher_logits = teacher_logits
        self.noisy_logits = noisy_logits
        self.labels = labels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        return (img,
                self.noisy_logits[idx],
                self.teacher_logits[idx],
                self.labels[idx])


@torch.no_grad()
def compute_teacher_logits(model, subset, batch_size, num_classes):
    """Compute reference logits f*(x_i) and labels on the TPV subset."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    N = len(subset)
    teacher_logits = torch.empty((N, num_classes), dtype=torch.float32)
    labels = torch.empty((N,), dtype=torch.long)
    model.eval()
    offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        bsz = x.size(0)
        z = model(x)
        teacher_logits[offset:offset+bsz].copy_(z.cpu())
        labels[offset:offset+bsz].copy_(y)
        offset += bsz
    return teacher_logits, labels


def _train_noisy(model, loader, max_epochs, lr):
    """MSE fine-tune on noisy logit targets. Runs in eval() mode to freeze BN."""
    model.eval()  # keep BN running stats frozen during fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.MSELoss()
    for _ in range(max_epochs):
        for x, noisy_z, _teacher_z, _y in loader:
            x = x.to(device, non_blocking=True)
            noisy_z = noisy_z.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), noisy_z)
            if not torch.isfinite(loss):
                return False
            loss.backward()
            optimizer.step()
    return True


def estimate_tpv(base_state_dict, num_classes, R, noise_std,
                 max_epochs_noisy, lr_noisy, train_batch_size, eval_batch_size,
                 model_factory):
    """
    Estimate empirical training-set TPV at the current model weights.

    Returns: float, mean squared deviation of fine-tuned logits from teacher
             logits, averaged over the TPV subset and R runs.
    """
    # 1. Reference (teacher) logits on TPV subset.
    teacher_model = model_factory().to(device)
    teacher_model.load_state_dict(base_state_dict)
    teacher_model.eval()
    teacher_logits, labels = compute_teacher_logits(
        teacher_model, tpv_subset, batch_size=eval_batch_size,
        num_classes=num_classes)
    del teacher_model
    torch.cuda.empty_cache()

    # 2. Adaptive noise scale: sigma * mean(|f*|).
    if ADAPTIVE_NOISE_SCALING:
        logit_scale = teacher_logits.abs().mean().item()
    else:
        logit_scale = 1.0

    # Deterministic shuffle order across runs so only logit noise varies.
    LOADER_SEED = 12345

    total_sqdiff = 0.0
    n_subset = teacher_logits.shape[0]

    for r in range(R):
        eps = torch.randn_like(teacher_logits) * noise_std * logit_scale
        noisy_logits = teacher_logits + eps

        run_dataset = _LogitNoiseDataset(
            subset=tpv_subset,
            teacher_logits=teacher_logits,
            noisy_logits=noisy_logits,
            labels=labels,
        )
        gen = torch.Generator(); gen.manual_seed(LOADER_SEED)
        run_loader = DataLoader(
            run_dataset, batch_size=train_batch_size, shuffle=True,
            generator=gen, num_workers=4, pin_memory=True)

        # Re-instantiate model from current checkpoint for this run.
        model = model_factory().to(device)
        model.load_state_dict(base_state_dict)

        ok = _train_noisy(model, run_loader,
                          max_epochs=max_epochs_noisy, lr=lr_noisy)
        if not ok:
            print(f"      [WARNING] TPV run {r} diverged, skipping.")
            del model; torch.cuda.empty_cache()
            continue

        # Squared deviation from teacher logits over the TPV subset.
        eval_loader = DataLoader(
            run_dataset, batch_size=eval_batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        run_sq = 0.0
        model.eval()
        with torch.no_grad():
            for x, _noisy_z, teacher_z, _y in eval_loader:
                x = x.to(device, non_blocking=True)
                teacher_z = teacher_z.to(device, non_blocking=True)
                z = model(x)
                run_sq += torch.sum((z - teacher_z) ** 2).item()
        total_sqdiff += run_sq

        del model
        torch.cuda.empty_cache()

    empirical_tpv = total_sqdiff / (R * n_subset)
    return empirical_tpv


# ----------------------------
# Build model + optimizer for the main training run
# ----------------------------
def make_model():
    return CIFARResNet18(num_classes=100)


model = make_model().to(device)
print(f"Model: CIFAR-ResNet-18 (params={count_params(model):,})")

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.wd, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=args.labelsmooth)

results_path = f"results/tpv_trajectory_cifar100_seed{args.seed}_noise_ratio-{args.noise_ratio}_labelsmooth-{args.labelsmooth}.pkl"

# ----------------------------
# Training loop with periodic TPV
# ----------------------------
epoch_log     = []
train_acc_log = []
val_acc_log   = []
tpv_epochs    = []
tpv_log       = []


def maybe_compute_tpv(epoch):
    """Compute TPV at this epoch and append to logs."""
    print(f"  [TPV] Estimating training-set TPV at epoch {epoch}...")
    base_state = copy.deepcopy(model.state_dict())
    tpv_value = estimate_tpv(
        base_state_dict=base_state,
        num_classes=100,
        R=args.R,
        noise_std=args.noise_std,
        max_epochs_noisy=args.max_epochs_noisy,
        lr_noisy=args.lr_noisy,
        train_batch_size=args.tpv_batch_size,
        eval_batch_size=args.tpv_batch_size,
        model_factory=make_model,
    )
    tpv_epochs.append(epoch)
    tpv_log.append(tpv_value)
    print(f"  [TPV] epoch {epoch}: TPV(train) = {tpv_value:.4e}")


# Compute TPV at epoch 0 (initial weights) for reference.
print("\n=== Computing TPV at epoch 0 (initialization) ===")
maybe_compute_tpv(epoch=0)

# Evaluate accuracy at epoch 0 too, so plots line up.
tr_acc0 = eval_accuracy(model, train_loader_aug)
va_acc0 = eval_accuracy(model, val_loader)
epoch_log.append(0)
train_acc_log.append(tr_acc0)
val_acc_log.append(va_acc0)
print(f"  Epoch 0: train_acc={tr_acc0:.4f}, val_acc={va_acc0:.4f}")

print("\n=== Training ===")
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader_aug,
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / max(n_batches,1):.3f}"})
    scheduler.step()

    train_acc = eval_accuracy(model, train_loader_aug)
    val_acc   = eval_accuracy(model, val_loader)
    epoch_log.append(epoch)
    train_acc_log.append(train_acc)
    val_acc_log.append(val_acc)
    print(f"  Epoch {epoch:3d} | lr={scheduler.get_last_lr()[0]:.4f} | "
          f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    if epoch % args.tpv_every == 0 or epoch == args.epochs:
        maybe_compute_tpv(epoch=epoch)

        # Persist intermediate logs.
        with open(results_path, "wb") as f:
            pkl.dump(dict(
                epoch_log=epoch_log,
                train_acc_log=train_acc_log,
                val_acc_log=val_acc_log,
                tpv_epochs=tpv_epochs,
                tpv_log=tpv_log,
                args=vars(args),
            ), f)

print(f"\nSaved logs to {results_path}")

# load results from results_path
with open(results_path, "rb") as f:
    data = pkl.load(f)
    epoch_log = data["epoch_log"]
    train_acc_log = data["train_acc_log"]
    val_acc_log = data["val_acc_log"]
    tpv_epochs = data["tpv_epochs"]
    tpv_log = data["tpv_log"]
# ----------------------------
# Plot: dual-axis (accuracy on left, TPV on right)
# ----------------------------
LABEL_FONTSIZE  = 16
TICK_FONTSIZE   = 13
LEGEND_FONTSIZE = 10

fig, ax_acc = plt.subplots(figsize=(6, 3))
ax_tpv = ax_acc.twinx()

l1, = ax_acc.plot(epoch_log, train_acc_log, "-",  color="tab:blue",
                  label="Train accuracy", linewidth=1.8)
l2, = ax_acc.plot(epoch_log, val_acc_log,   "-",  color="tab:orange",
                  label="Validation accuracy", linewidth=1.8)
l3, = ax_tpv.plot(tpv_epochs, tpv_log,      "-o", color="tab:green",
                  label="Training-set TPV", linewidth=1.8, markersize=5)

ax_acc.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
ax_acc.set_ylabel("Accuracy", fontsize=LABEL_FONTSIZE, color="black")
ax_tpv.set_ylabel("Training-set TPV", fontsize=LABEL_FONTSIZE,
                  color="tab:green")
ax_acc.tick_params(labelsize=TICK_FONTSIZE)
ax_tpv.tick_params(labelsize=TICK_FONTSIZE, colors="tab:green")
ax_tpv.set_yscale("log")

# Mark the epoch with the highest validation accuracy and the epoch with the
best_val_epoch = int(np.argmax(val_acc_log))
l4 = ax_acc.axvline(5*epoch_log[int(np.argmax(tpv_log))], linestyle="--", color="tab:green",
               alpha=0.4, label=f"argmax_epoch TPV")

# For TPV minimum, ignore epoch 0 (initialization) where TPV may be tiny
# because the random model produces near-zero logits.
if len(tpv_log) > 1:
    post_warmup = [(e, t) for e, t in zip(tpv_epochs, tpv_log) if e > 0]
    if post_warmup:
        es = [e for e, _ in post_warmup]
        ts = [t for _, t in post_warmup]
        min_tpv_epoch = es[int(np.argmin(ts))]

# Combine legends from both axes.
lines  = [l1, l2, l3, l4]
labels = [ln.get_label() for ln in lines]

ax_acc.legend(lines, labels, fontsize=LEGEND_FONTSIZE,
              loc="lower right", framealpha=0.4)

plt.tight_layout()
plot_path = f"results/plots/tpv_trajectory_cifar100_seed{args.seed}_noise_ratio-{args.noise_ratio}_labelsmooth-{args.labelsmooth}.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.show()
print(f"Saved plot to: {plot_path}")
