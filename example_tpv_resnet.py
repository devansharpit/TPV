"""
Example: compute TPV for all three noise sources on a pretrained ResNet (CIFAR-10).

Usage:
    python example_tpv_resnet.py

Loads ResNet-20 pretrained on CIFAR-10 (smallest standard CIFAR ResNet available
via the chenyaofo hub, analogous to ResNet-18 for ImageNet). Computes:
  - SGD noise TPV    via Hutchinson trace estimator
  - Quantization TPV via Hutchinson trace estimator
  - Label noise TPV  via Monte Carlo empirical estimation
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tpv.sgd_noise.sgd_tpv import SgdTPV
from tpv.quantization_noise.quantization_tpv import QuantizationTPV
from tpv.label_noise.label_tpv import LabelTPV

# ---------------------------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "cifar10_resnet20"   # pretrained ResNet-20 from chenyaofo hub
N_TRAIN = 2000                    # training subset size
N_TEST = 2000                     # test subset size

# SGD noise TPV
SGD_LR = 1e-3
SGD_BATCH_SIZE = 128
N_HUTCHINSON = 50                 # Rademacher vectors for trace estimator

# Quantization noise TPV
QUANT_BITS = 8

# Label noise TPV
LABEL_NOISE_STD = 0.1
LABEL_R = 5                       # Monte Carlo runs
LABEL_MAX_EPOCHS = 5
LABEL_LR = 1e-4
LABEL_BATCH_SIZE = 256

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# ---------------------------------------------------------------------------
# Data — small CIFAR-10 subsets loaded as in-memory tensors
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_indices = np.random.choice(len(train_dataset), N_TRAIN, replace=False)
test_indices = np.random.choice(len(test_dataset), N_TEST, replace=False)

X_train = torch.stack([train_dataset[i][0] for i in train_indices]).to(device)
y_train = torch.tensor([train_dataset[i][1] for i in train_indices], dtype=torch.long, device=device)

X_test = torch.stack([test_dataset[i][0] for i in test_indices]).to(device)
y_test = torch.tensor([test_dataset[i][1] for i in test_indices], dtype=torch.long, device=device)

print(f"Train subset: {X_train.shape}, Test subset: {X_test.shape}")

# ---------------------------------------------------------------------------
# Pretrained model
# ---------------------------------------------------------------------------
print(f"\nLoading pretrained {MODEL_NAME}...")
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    MODEL_NAME,
    pretrained=True,
    verbose=False,
).to(device)
model.eval()

with torch.no_grad():
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
print(f"Clean model test accuracy (subset): {acc:.3f}")

base_state_dict = copy.deepcopy(model.state_dict())

# ---------------------------------------------------------------------------
# 1. SGD noise TPV
# ---------------------------------------------------------------------------
print(f"\n--- SGD noise TPV (trace form; lr={SGD_LR}, batch_size={SGD_BATCH_SIZE}, "
      f"n_hutchinson={N_HUTCHINSON}) ---")

sgd_tpv = SgdTPV(device=device, seed=SEED)
tpv_sgd_train, trace_H_sgd_train = sgd_tpv.compute_tpv(
    model, X_train, y_train,
    lr=SGD_LR,
    batch_size=SGD_BATCH_SIZE,
    n_hutchinson_samples=N_HUTCHINSON,
)

tpv_sgd_test, trace_H_sgd_test = sgd_tpv.compute_tpv(
    model, X_test, y_test,
    lr=SGD_LR,
    batch_size=SGD_BATCH_SIZE,
    n_hutchinson_samples=N_HUTCHINSON,
)

print(f"  Tr(H_eff) train     = {trace_H_sgd_train:.4f}")
print(f"  TPV_sgd_train     = {tpv_sgd_train:.6e}  "
      f"[= lr/(2*bs) * Tr(H) = {SGD_LR/(2*SGD_BATCH_SIZE):.2e} * {trace_H_sgd_train:.4f}]")

print(f"  Tr(H_eff) test     = {trace_H_sgd_test:.4f}")
print(f"  TPV_sgd_test     = {tpv_sgd_test:.6e}  "
      f"[= lr/(2*bs) * Tr(H) = {SGD_LR/(2*SGD_BATCH_SIZE):.2e} * {trace_H_sgd_test:.4f}]")

# ---------------------------------------------------------------------------
# 1b. SGD noise TPV — empirical trajectory variant
# ---------------------------------------------------------------------------
SGD_EMP_STEPS = 200
SGD_EMP_BURN_IN = 50
SGD_EMP_SNAPSHOT_EVERY = 10

print(f"\n--- SGD noise TPV (empirical trajectory; lr={SGD_LR}, "
      f"batch_size={SGD_BATCH_SIZE}, sgd_steps={SGD_EMP_STEPS}) ---")

emp_sgd = sgd_tpv.compute_tpv_empirical(
    model, X_train, y_train,
    X_test=X_test,
    lr=SGD_LR,
    batch_size=SGD_BATCH_SIZE,
    sgd_steps=SGD_EMP_STEPS,
    burn_in=SGD_EMP_BURN_IN,
    snapshot_every=SGD_EMP_SNAPSHOT_EVERY,
)

print(f"  TPV_sgd_emp_train = {emp_sgd['tpv_train']:.6e}")
print(f"  TPV_sgd_emp_test  = {emp_sgd['tpv_test']:.6e}")
print(f"  ({emp_sgd['n_snapshots']} snapshots)")

# ---------------------------------------------------------------------------
# 2. Quantization noise TPV — analytical (trace form)
# ---------------------------------------------------------------------------
print(f"\n--- Quantization noise TPV (trace form; {QUANT_BITS}-bit uniform, "
      f"n_hutchinson={N_HUTCHINSON}) ---")

quant_tpv = QuantizationTPV(device=device, seed=SEED)
tpv_quant_train, trace_H_quant_train = quant_tpv.compute_tpv(
    model, X_train, y_train,
    n_bits=QUANT_BITS,
    n_hutchinson_samples=N_HUTCHINSON,
)

tpv_quant_test, trace_H_quant_test = quant_tpv.compute_tpv(
    model, X_test, y_test,
    n_bits=QUANT_BITS,
    n_hutchinson_samples=N_HUTCHINSON,
)

all_w = torch.cat([p.detach().flatten() for p in model.parameters()])
w_range = all_w.max().item() - all_w.min().item()
delta = w_range / (2 ** QUANT_BITS - 1)
sigma_q_sq = delta ** 2 / 12.0
print(f"  weight range = {w_range:.4f}, delta = {delta:.6f}, sigma_q² = {sigma_q_sq:.2e}")
print(f"  Tr(H_eff) train     = {trace_H_quant_train:.4f}")
print(f"  TPV_quant_train   = {tpv_quant_train:.6e}  [= sigma_q² * Tr(H)]")
print(f"  Tr(H_eff) test      = {trace_H_quant_test:.4f}")
print(f"  TPV_quant_test    = {tpv_quant_test:.6e}  [= sigma_q² * Tr(H)]")

# ---------------------------------------------------------------------------
# 2b. Quantization noise TPV — empirical Monte Carlo variant
# ---------------------------------------------------------------------------
QUANT_EMP_RUNS = 50

print(f"\n--- Quantization noise TPV (empirical Monte Carlo; "
      f"{QUANT_BITS}-bit uniform, n_runs={QUANT_EMP_RUNS}) ---")

emp_quant = quant_tpv.compute_tpv_empirical(
    model, X_train, X_test=X_test,
    n_bits=QUANT_BITS,
    n_runs=QUANT_EMP_RUNS,
    seed=SEED,
)
print(f"  TPV_quant_emp_train = {emp_quant['tpv_train']:.6e}")
print(f"  TPV_quant_emp_test  = {emp_quant['tpv_test']:.6e}")
print(f"  ({emp_quant['n_runs']} runs)")


# ---------------------------------------------------------------------------
# 3. Label noise TPV
# ---------------------------------------------------------------------------
print(f"\n--- Label noise TPV (noise_std={LABEL_NOISE_STD}, R={LABEL_R}, "
      f"epochs={LABEL_MAX_EPOCHS}) ---")


def train_fn(m, X, y_noisy):
    """Fine-tune m to regress logits to y_noisy via mini-batch SGD with MSE."""
    m.eval()  # keep BN/Dropout frozen; only weights change
    optimizer = optim.SGD(m.parameters(), lr=LABEL_LR, momentum=0.9, weight_decay=0)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, y_noisy)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=LABEL_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(12345),
    )
    for _ in range(LABEL_MAX_EPOCHS):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(m(bx), by).backward()
            optimizer.step()


def model_factory():
    return torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        MODEL_NAME,
        pretrained=False,
        verbose=False,
    )


label_tpv = LabelTPV(device=device, seed=SEED)
tpv_label = label_tpv.compute_tpv(
    model_factory=model_factory,
    base_state_dict=base_state_dict,
    X_train=X_train,
    X_test=X_test,
    noise_std=LABEL_NOISE_STD,
    R=LABEL_R,
    train_fn=train_fn,
)
print(f"  TPV_label_train = {tpv_label['empirical_TPV_train']:.6e}")
print(f"  TPV_label_test  = {tpv_label['empirical_TPV_test']:.6e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _fmt(v):
    return f"{v:.4e}"

def _ratio(a, b):
    return f"{a/b:.3f}" if b else "  n/a"

_rows = [
    ("SGD noise",    "trace form", tpv_sgd_train,         tpv_sgd_test,
        f"lr={SGD_LR}, batch={SGD_BATCH_SIZE}"),
    ("SGD noise",    "empirical",  emp_sgd['tpv_train'],  emp_sgd['tpv_test'],
        f"sgd_steps={SGD_EMP_STEPS}, snaps={emp_sgd['n_snapshots']}"),
    ("Quantization", "trace form", tpv_quant_train,       tpv_quant_test,
        f"{QUANT_BITS}-bit uniform"),
    ("Quantization", "empirical",  emp_quant['tpv_train'], emp_quant['tpv_test'],
        f"{QUANT_BITS}-bit uniform, n_runs={emp_quant['n_runs']}"),
    ("Label noise",  "empirical",  tpv_label['empirical_TPV_train'],
        tpv_label['empirical_TPV_test'],
        f"noise_std={LABEL_NOISE_STD}, R={LABEL_R}"),
]

print()
print("=" * 88)
print(f"  TPV Summary — {MODEL_NAME} (pretrained on CIFAR-10)")
print("=" * 88)
print(f"  {'Perturbation':<14}{'Estimator':<14}"
      f"{'Train TPV':>14}{'Test TPV':>14}  {'Notes'}")
print("  " + "-" * 86)
for src, est, tr, te, note in _rows:
    print(f"  {src:<14}{est:<14}{_fmt(tr):>14}{_fmt(te):>14}  {note}")
print("=" * 88)

print()
print("  Train/Test stability (TPV_train / TPV_test; should ideally be close to 1):")
for src, est, tr, te, _ in _rows:
    print(f"    {src:<14} {est:<12} {_ratio(tr, te):>8}")