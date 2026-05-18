import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
TPV Multi-regularizer joint configurations on CIFAR-10
=============================================================
Experimental design:
  - ``N_CONFIGS`` distinct regularization configurations are sampled
    once.  Each configuration specifies a (weight_decay, dropout_p,
    label_smoothing) triple, drawn jointly from candidate level grids
    for each regularizer.
  - The SAME ``N_CONFIGS`` configurations are used to train every
    architecture size (each "group" in GROUP_CONFIGS).  This groups each
    (group, config) cell across the full architecture sweep.
  - For each (group, config), we train a reference MLP to convergence
    on CIFAR-10, then estimate TPV via noisy-logit fine-tuning.

Purpose:
  Earlier experiments varied a single regularizer at a time.  A valid
  skeptical reading of those plots is that TPV merely tracks the
  regularizer strength, so a practitioner could rank models by
  regularizer strength directly and never compute TPV.  This
  experiment tests whether TPV adds information *beyond* any single
  regularizer's strength by using multiple regularizers simultaneously
  at independently sampled levels.  The configurations are not ordered
  by any single axis, so TPV's ranking cannot be read off a
  regularizer value.

Usage:
    python tpv_cifar_reg_combination.py
"""

import numpy as np
import pickle as pkl
import random
import copy
import argparse
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from tpv.label_noise import LabelTPV

# ----------------------------
# Argument parsing
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="TPV MLP multi-regularizer joint configuration experiment."
    )
    parser.add_argument(
        "--n_configs", type=int, default=5,
        help="Number of joint (wd, dropout, labelsmooth) configurations to sample.",
    )
    parser.add_argument(
        "--arch", type=str, default="resnet", choices=["mlp", "resnet"],
        help="Architecture type to use: 'mlp' or 'resnet'.",
    )
    parser.add_argument(
        "--config_seed", type=int, default=0,
        help="RNG seed for sampling the joint regularization configurations.",
    )
    return parser.parse_args()

args = get_args()
N_CONFIGS   = args.n_configs
CONFIG_SEED = args.config_seed
ARCH        = args.arch

# Candidate level grids per regularizer.  Zero is included in every grid
# so "no regularization on this axis" is a possible outcome of sampling.
WD_CANDIDATES          = [0.0, 1e-5, 5e-4]
DROPOUT_CANDIDATES     = [0.0, 0.1,  0.2,  0.3,  0.5]
LABELSMOOTH_CANDIDATES = [0.0, 0.05, 0.1,  0.2,  0.3]

# Sample N_CONFIGS joint configurations.  Each sample draws one level
# from each candidate list independently.  Seed is fixed so all groups
# see the same configs.
_cfg_rng = np.random.default_rng(CONFIG_SEED)
REG_CONFIGS = []
for ci in range(N_CONFIGS):
    REG_CONFIGS.append(dict(
        config_id=ci,
        wd         = float(_cfg_rng.choice(WD_CANDIDATES)),
        dropout    = float(_cfg_rng.choice(DROPOUT_CANDIDATES)),
        labelsmooth= float(_cfg_rng.choice(LABELSMOOTH_CANDIDATES)),
    ))

def config_short_label(c):
    """Compact label for legend: e.g. 'C0: wd=1e-03, d=0.1, ε=0.05'."""
    wd = c["wd"]; dp = c["dropout"]; ls = c["labelsmooth"]
    wd_s  = f"wd={wd:.0e}" if wd  > 0 else "wd=0"
    dp_s  = f"dropout={dp:g}"
    ls_s  = f"labelsmooth={ls:g}"
    return f"C{c['config_id']}: {wd_s}, {dp_s}, {ls_s}"

print("\nSampled regularization configurations:")
for c in REG_CONFIGS:
    print(f"  {config_short_label(c)}")

# ----------------------------
# Reproducibility & device
# ----------------------------
GLOBAL_SEED = 0
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("./results/", exist_ok=True)
os.makedirs("./results/plots/", exist_ok=True)


# ----------------------------
# MLP architecture
# ----------------------------
class MLP(nn.Module):
    """
    2-hidden-layer ReLU MLP with optional dropout after each hidden layer.
    dropout_p=0.0 means no dropout (standard MLP).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 dropout_p=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim),  nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class _ResBasicBlock(nn.Module):
    """Basic residual block: two 3×3 convs with optional dropout."""
    def __init__(self, in_ch, out_ch, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


# Mapping from num_layers → (blocks per stage) for the 4-stage ResNet.
# Covers standard ResNet-18 / 34 depths; other values fall back to 18.
_RESNET_STAGE_BLOCKS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
}

class ConvResNet(nn.Module):
    """
    CIFAR-adapted ResNet with controllable width (hidden_dim) and depth
    (num_layers, default 18).  Constructor arguments mirror MLP so that
    both architectures can be driven by the same config dict.

    hidden_dim  – base channel width; stages use [C, 2C, 4C, 8C]
    num_layers  – target depth: 18 → [2,2,2,2] blocks, 34 → [3,4,6,3]
    num_classes – output classes
    dropout_p   – Dropout2d probability inside each residual block
    input_dim   – accepted for API compatibility; in_channels inferred as
                  input_dim // (32*32) (equals 3 for CIFAR-10/100)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes,
                 dropout_p=0.0):
        super().__init__()
        in_ch   = max(1, input_dim // (32 * 32))   # 3 for CIFAR
        C       = hidden_dim
        blocks  = _RESNET_STAGE_BLOCKS.get(num_layers, _RESNET_STAGE_BLOCKS[18])

        # Stem: single 3×3 conv (CIFAR-style; no max-pool to preserve 32×32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Four residual stages; spatial resolution halved at stages 2-4
        self.stage1 = self._make_stage(C,   C,   blocks[0], stride=1, dp=dropout_p)
        self.stage2 = self._make_stage(C,   C*2, blocks[1], stride=2, dp=dropout_p)
        self.stage3 = self._make_stage(C*2, C*4, blocks[2], stride=2, dp=dropout_p)
        self.stage4 = self._make_stage(C*4, C*8, blocks[3], stride=2, dp=dropout_p)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(C * 8, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, stride, dp):
        layers = [_ResBasicBlock(in_ch, out_ch, stride=stride, dropout_p=dp)]
        for _ in range(num_blocks - 1):
            layers.append(_ResBasicBlock(out_ch, out_ch, stride=1, dropout_p=dp))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Accept both image tensors (N,C,H,W) and flat vectors (N, C*H*W)
        if x.dim() == 2:
            in_ch = max(1, x.size(1) // (32 * 32))
            x = x.view(x.size(0), in_ch, 32, 32)
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_network(hidden_dim, num_layers, dropout_p=0.0,
             num_classes=10, input_dim=3072, arch=ARCH):
    if arch == "mlp":
        return MLP(input_dim, hidden_dim, num_layers, num_classes,
                   dropout_p=dropout_p).to(device)
    elif arch == "resnet":
        return ConvResNet(input_dim, hidden_dim, num_layers, num_classes,
                          dropout_p=dropout_p).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ----------------------------
# Group definitions
# ----------------------------
NUM_CLASSES = 10
INPUT_DIM   = 3 * 32 * 32


# (hidden_dim, group_label) — p increases across groups by ~2 OOM

if args.arch == "mlp":
    NUM_LAYERS  = 2   # fixed depth across all groups
    GROUP_CONFIGS = [
        [128,  "Group 2"],   # p ≈ 120k
        [256,  "Group 3"],   # p ≈ 400k
        [512,  "Group 4"],   # p ≈ 1.3M
        [1024, "Group 5"],   # p ≈ 4.2M
    ]
elif args.arch == "resnet":
    NUM_LAYERS  = 18
    GROUP_CONFIGS = [
        [8,  "Group 2"],   # p ≈ 176k
        [16,  "Group 3"],   # p ≈ 701k
        [24,  "Group 4"],   # p ≈ 1.5M
        [32, "Group 5"],   # p ≈ 2.8M
    ]

# Print parameter counts (dropout_p=0 for counting — dropout adds no params)
print("\n=== Parameter counts per group ===")
for (hdim, plabel) in GROUP_CONFIGS:
    m = make_network(hdim, NUM_LAYERS, dropout_p=0.0)
    print(f"{plabel}: hidden={hdim}, p={count_params(m):,}")
    del m

# Build flat list of (group × config) model configs.
model_configs = []
param_cnts = []
for (hdim, plabel) in GROUP_CONFIGS:
    # Use dropout_p=0 for param count since dropout adds no parameters
    p = count_params(make_network(hdim, NUM_LAYERS, dropout_p=0.0))
    param_cnts.append(p)
    for rc in REG_CONFIGS:
        model_configs.append(dict(
            group_label=f'p={p//1000}k',
            config_id=rc["config_id"],
            role=f"config_{rc['config_id']}",
            wd=rc["wd"],
            dropout=rc["dropout"],
            labelsmooth=rc["labelsmooth"],
            hidden_dim=hdim,
            num_layers=NUM_LAYERS,
            num_params=p,
        ))
n_models = len(model_configs)
for i in range(len(param_cnts)):
    GROUP_CONFIGS[i][1] = f"p={param_cnts[i]//1000}k"
print(f"\nTotal models: {n_models}  "
      f"({len(GROUP_CONFIGS)} groups × {N_CONFIGS} joint configs)")


# ----------------------------
# Training utilities
# ----------------------------
def train_clean_model(model, X_train, y_train, X_test, y_test,
                      max_epochs=300, lr=1e-2, batch_size=256,
                      momentum=0.9,
                      wd=0.0, labelsmooth=0.0):
    """
    CE trainer with cosine LR schedule.  All three regularizers are
    applied simultaneously:
      - ``wd``          : SGD weight_decay
      - ``labelsmooth`` : nn.CrossEntropyLoss(label_smoothing=...)
      - dropout         : already baked into the model architecture at
                          construction time (dropout_p passed to make_network),
                          so there is nothing for this function to do.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=wd, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max_epochs//3, gamma=0.1)

    criterion = nn.CrossEntropyLoss(label_smoothing=labelsmooth)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(max_epochs),
                desc=f"wd={wd:.0e},ls={labelsmooth:g}", leave=False)
    for epoch in pbar:
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Always eval CE without label smoothing for comparability
                ce_crit = nn.CrossEntropyLoss()
                ce_tr = ce_crit(model(X_train), y_train).item()
                ce_te = ce_crit(model(X_test),  y_test).item()
            pbar.set_postfix({"tr": f"{ce_tr:.3f}", "te": f"{ce_te:.3f}"})
    model.eval()


def eval_ce(model, X, y, batch_size=512):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            total += criterion(model(X[i:i+batch_size]),
                               y[i:i+batch_size]).item()
    return total / X.shape[0]


# ----------------------------
# Data loading
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])
n_train_sub, n_test_sub = 10000, 4000
rng = np.random.default_rng(0)

def load_subset(ds, idx):
    Xs, ys = [], []
    for i in idx:
        x, y = ds[i]; Xs.append(x); ys.append(y)
    return (torch.stack(Xs).to(device),
            torch.tensor(ys, dtype=torch.long, device=device))

train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
X_train, y_train = load_subset(train_ds, rng.choice(len(train_ds), n_train_sub, replace=False))
X_test,  y_test  = load_subset(test_ds,  rng.choice(len(test_ds),  n_test_sub,  replace=False))
print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test.")


# ----------------------------
# Experiment config
# ----------------------------
noise_std        = 0.1
R                = 20
max_epochs_clean = 300
max_epochs_noisy = 20
lr_clean         = 5e-2
lr_noisy         = 1e-4
batch_size       = 256
proximity_lambda = 0.0
fname = f"results/tpv_mlp_groups_cifar10_multireg_n{N_CONFIGS}_seed{CONFIG_SEED}.pkl"

# If ``force_train`` is False and a cached results file exists, we skip
# training and just re-load + re-plot.  Set to True to always retrain.
force_train = True


# ----------------------------
# Main loop
# ----------------------------
if force_train or not os.path.exists(fname):
    all_results = []
    for mi, cfg in enumerate(model_configs):
        print(f"\n=== [{mi+1}/{n_models}] {cfg['group_label']} — config C{cfg['config_id']} "
              f"(h={cfg['hidden_dim']}, wd={cfg['wd']:.0e}, "
              f"dropout={cfg['dropout']:g}, eps={cfg['labelsmooth']:g}, "
              f"p={cfg['num_params']:,}) ===")

        # Dropout is baked into the architecture at construction time.
        dropout_p = cfg["dropout"]
        model_clean = make_network(cfg["hidden_dim"], cfg["num_layers"],
                               dropout_p=dropout_p)

        train_clean_model(
            model_clean, X_train, y_train, X_test, y_test,
            max_epochs=max_epochs_clean, lr=lr_clean,
            batch_size=batch_size, momentum=0.9,
            wd=cfg["wd"], labelsmooth=cfg["labelsmooth"],
        )

        # Always evaluate CE without label smoothing / dropout for clean comparison
        model_clean.eval()
        ce_train_ref = eval_ce(model_clean, X_train, y_train)
        ce_test_ref  = eval_ce(model_clean, X_test,  y_test)
        print(f"  Ref CE — train: {ce_train_ref:.4f}  test: {ce_test_ref:.4f}")

        base_state_dict = copy.deepcopy(model_clean.state_dict())

        def model_factory():
            m = make_network(cfg["hidden_dim"], cfg["num_layers"], dropout_p=dropout_p)
            return m.to(device)

        def train_fn(m, X, y_noisy):
            m.eval()
            optimizer = optim.SGD(m.parameters(), lr=lr_noisy, momentum=0.9, weight_decay=0.0)
            criterion = nn.MSELoss()
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y_noisy),
                batch_size=batch_size, shuffle=False,
                generator=torch.Generator().manual_seed(12345))
            for _ in range(max_epochs_noisy):
                for bx, by in loader:
                    optimizer.zero_grad()
                    criterion(m(bx), by).backward()
                    optimizer.step()

        tpv = LabelTPV(device=device, seed=GLOBAL_SEED).compute_tpv(
            model_factory=model_factory,
            base_state_dict=base_state_dict,
            X_train=X_train,
            X_test=X_test,
            noise_std=noise_std,
            R=R,
            train_fn=train_fn,
        )

        all_results.append(dict(**cfg, ce_train_ref=ce_train_ref,
                                ce_test_ref=ce_test_ref, **tpv))
        print(f"  TPV (test): {tpv['empirical_TPV_test']:.4e}")

        # Save incrementally so long runs can be resumed if interrupted.
        with open(fname, "wb") as f:
            pkl.dump(all_results, f)

    print(f"\nSaved to {fname}")

# ----------------------------
# Plotting
# ----------------------------
results     = pkl.load(open(fname, "rb"))
group_labels = [cfg[1] for cfg in GROUP_CONFIGS]
group_colors = plt.cm.tab10(np.linspace(0, 0.5, len(group_labels)))
color_map    = dict(zip(group_labels, group_colors))

# One marker per joint configuration (config_id = 0, 1, ..., N_CONFIGS-1)
all_markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
marker_map  = {f"config_{i}": all_markers[i % len(all_markers)]
               for i in range(N_CONFIGS)}

# Human-readable config labels for the legend (one per sampled joint config)
config_labels = {
    f"config_{c['config_id']}": config_short_label(c)
    for c in REG_CONFIGS
}

tpv  = np.array([r["empirical_TPV_test"]  for r in results])
tpv_train = np.array([r["empirical_TPV_train"] for r in results])
ce_test  = np.array([r["ce_test_ref"]   for r in results])

rho,  _ = stats.spearmanr(tpv,  ce_test)
print(f"\nSpearman(TPV, CE test) = {rho:.3f}")

LABEL_FONTSIZE  = 20
TICK_FONTSIZE   = 20
TITLE_FONTSIZE  = 15
LEGEND_FONTSIZE = 12

for tr, te in zip(tpv, tpv_train):
    print(f"TPV test: {tr:.4e}  TPV train: {te:.4e}")

# -------------------------------------------------------
# Figure 1 (KEY FIGURE): two-panel scatter
# -------------------------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(9, 5))

for ax, (tpv_vals, tpv_vals_train, xlabel, rho, title) in zip([axes], [
    (tpv, tpv_train, "TPV (train)",
     rho,  f"TPV vs Test CE\n(Spearman $\\rho={rho:.2f}$)"),
]):
    ax2 = ax.twinx()
    for r in results:
        xval = r["empirical_TPV_test"]
        ax.scatter(xval, r["ce_test_ref"],
                   color=color_map[r["group_label"]],
                   marker=marker_map[r["role"]],
                   s=120, zorder=3, linewidths=0.5, edgecolors="k")
        # create twin x-axis and plot training CE on it whose values are shown on right y-axis

        ax2.scatter(xval, r["ce_train_ref"],
                    color=color_map[r["group_label"]],
                    marker=marker_map[r["role"]],
                    s=50, zorder=3, linewidths=0.2, edgecolors="k", alpha=0.4)

    # Connect same-color (same-group) markers with dashed lines, sorted by TPV
    for pl in group_labels:
        sub = [r for r in results if r["group_label"] == pl]
        sub.sort(key=lambda r: r["empirical_TPV_test"])
        xs_line = [r["empirical_TPV_test"] for r in sub]
        ys_line = [r["ce_test_ref"] for r in sub]
        ax.plot(xs_line, ys_line, "--", color=color_map[pl],
                linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_xscale("log")
    log_x = np.log10(tpv_vals_train)
    coef  = np.polyfit(log_x, ce_test, 1)
    xs    = np.logspace(log_x.min(), log_x.max(), 200)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Test CE Loss (solid markers)", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Train CE Loss (faded markers)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

# Legend: groups (colour) + joint regularization configs (marker)
legend_elements = (
    [Line2D([0], [0], color=color_map[pl], lw=0, marker="o",
             markersize=8, label=pl) for pl in group_labels]
    + [Line2D([0], [0], color="k", lw=0, marker=marker_map[f"config_{i}"],
               markersize=8, label=config_labels[f"config_{i}"])
       for i in range(N_CONFIGS)]
)
# push legend a little lower to avoid overlapping with highest group
axes.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE,
               loc="upper left", framealpha=0, bbox_to_anchor=(0, 0.95))
plt.tight_layout()
plot_path = f"results/plots/multireg_tpv_n{N_CONFIGS}_seed{CONFIG_SEED}.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")


# -------------------------------------------------------
# Summary table
# -------------------------------------------------------
print(f"\n{'Model':<48} {'p':>8} {'CE train':>9} {'CE test':>9} {'TPV':>13}")
print("-" * 105)
for r in results:
    label = (f"{r['group_label']} C{r['config_id']} "
             f"(wd={r['wd']:.0e},d={r['dropout']:g},ls={r['labelsmooth']:g})")
    print(f"{label:<48} {r['num_params']:>8,} {r['ce_train_ref']:>9.4f} "
          f"{r['ce_test_ref']:>9.4f} "
          f"{r['empirical_TPV_test']:>13.4e}")
print(f"\nSpearman(TPV, CE test) = {rho:.3f}")
