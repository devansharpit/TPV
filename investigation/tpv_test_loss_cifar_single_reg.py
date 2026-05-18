import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
TPV MLP Experiment with models trained with varying regularization levels
=================================================
Experimental design:
  - 5 architecture sizes, each trained at N_LEVELS regularisation
    strengths of one chosen regularisation type (--reg_type flag):
        wd       : weight decay in SGD   (levels: 0, 1e-4, 5e-4, 1e-3)
        dropout  : dropout probability   (levels: 0, 0.1,  0.2,  0.3)
        labelsmooth: label smoothing ε   (levels: 0, 0.05, 0.1,  0.2)
  - Within each architecture, generalization varies only through regularisation strength.

Usage:
    python tpv_test_loss_cifar_single_reg.py --reg_type wd
    python tpv_test_loss_cifar_single_reg.py --reg_type dropout
    python tpv_test_loss_cifar_single_reg.py --reg_type labelsmooth
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

# ----------------------------
# Argument parsing
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="TPV MLP groups experiment.")
    parser.add_argument(
        "--reg_type", type=str, default="wd",
        choices=["wd", "dropout", "labelsmooth"],
        help=(
            "Regularisation type.\n"
            "  wd          : weight decay (SGD weight_decay param)\n"
            "  dropout     : dropout probability after each hidden layer\n"
            "  labelsmooth : label smoothing epsilon in cross-entropy loss"
        ),
    )
    parser.add_argument("--no_adaptive_noise_scaling", action="store_true")
    return parser.parse_args()

args = get_args()
REG_TYPE = args.reg_type
ADAPTIVE_NOISE_SCALING = not args.no_adaptive_noise_scaling
# Regularisation levels per type.
REG_LEVELS = {
    "wd":          [0, 1e-5, 1e-4, 1e-3, 1e-2,],
    "dropout":     [0.0,  0.1,  0.2,  0.3, 0.7,],
    "labelsmooth": [0.0,  0.05, 0.1,  0.2, 0.7, 0.9],
}
LEVELS = REG_LEVELS[REG_TYPE]
N_LEVELS = len(LEVELS)

print(f"\nRegularisation type : {REG_TYPE}")
print(f"Levels              : {LEVELS}")

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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_mlp(hidden_dim, num_layers, dropout_p=0.0,
             num_classes=10, input_dim=3072):
    return MLP(input_dim, hidden_dim, num_layers, num_classes,
               dropout_p=dropout_p).to(device)


# ----------------------------
# Group definitions
# ----------------------------
NUM_CLASSES = 10
INPUT_DIM   = 3 * 32 * 32
NUM_LAYERS  = 2   # fixed depth across all groups

# (hidden_dim, group_label) — p increases across groups by ~2 OOM
GROUP_CONFIGS = [
    [128,  "Group 2"],   # p ≈ 120k
    [256,  "Group 3"],   # p ≈ 400k
    [512,  "Group 4"],   # p ≈ 1.8M
    [1024, "Group 5"],   # p ≈ 4.2M
]

# Print parameter counts
print("\n=== Parameter counts ===")
for (hdim, glabel) in GROUP_CONFIGS:
    m = make_mlp(hdim, NUM_LAYERS, dropout_p=0.0)
    print(f"{glabel}: hidden={hdim}, p={count_params(m):,}")
    del m

# Build flat list of N_LEVELS * 5 model configs.
# role = "level_{i}" where i=0 is least regularised (best generalisation).
model_configs = []
param_cnts = []
for (hdim, glabel) in GROUP_CONFIGS:
    # Use dropout_p=0 for param count since dropout adds no parameters
    p = count_params(make_mlp(hdim, NUM_LAYERS, dropout_p=0.0))
    param_cnts.append(p)
    for i, level in enumerate(LEVELS):
        model_configs.append(dict(
            group_label=f'p={p//1000}k',
            role=f"level_{i}",
            reg_level=level,
            hidden_dim=hdim,
            num_layers=NUM_LAYERS,
            num_params=p,
        ))
n_models = len(model_configs)
for i in range(len(param_cnts)):
    GROUP_CONFIGS[i][1] = f"p={param_cnts[i]//1000}k"
print(f"\nTotal models: {n_models}  "
      f"({len(GROUP_CONFIGS)} groups × {N_LEVELS} {REG_TYPE} levels)")


# ----------------------------
# Training utilities
# ----------------------------
def compute_proximity_penalty(model, ref_state_dict):
    penalty = 0.0
    for name, param in model.named_parameters():
        if name in ref_state_dict:
            penalty = penalty + torch.sum((param - ref_state_dict[name]) ** 2)
    return penalty


def train_clean_model(model, X_train, y_train, X_test, y_test,
                      max_epochs=300, lr=1e-3, batch_size=256,
                      momentum=0.9, reg_type="wd", reg_level=0.0):
    """
    CE trainer with cosine LR schedule.
    reg_type / reg_level control which regularisation is applied:
      wd          → SGD weight_decay = reg_level
      dropout     → dropout already baked into model architecture;
                    reg_level stored for reference but not re-applied here
      labelsmooth → nn.CrossEntropyLoss(label_smoothing=reg_level)
    """
    wd = reg_level if reg_type == "wd" else 0.0
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max_epochs//3, gamma=0.1)

    label_smoothing = reg_level if reg_type == "labelsmooth" else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(max_epochs),
                desc=f"{reg_type}={reg_level:.0e}", leave=False)
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


def train_noisy_model(model, X_train, y_noisy,
                      max_epochs=20, lr=1e-4, batch_size=256,
                      ref_state_dict=None, proximity_lambda=0.0,
                      dataloader_seed=12345):
    """MSE regression on noisy logit targets."""
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.MSELoss()
    gen = torch.Generator(); gen.manual_seed(dataloader_seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_noisy),
        batch_size=batch_size, shuffle=False, generator=gen)
    for _ in range(max_epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            if ref_state_dict is not None and proximity_lambda > 0:
                loss = loss + proximity_lambda * compute_proximity_penalty(
                    model, ref_state_dict)
            if not torch.isfinite(loss):
                return False
            loss.backward()
            optimizer.step()
    return True


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
# TPV estimation
# ----------------------------
def estimate_tpv(cfg, noise_std, R,
                 X_train, y_train, X_test, y_test,
                 base_state_dict, f_star_train, f_star_test,
                 max_epochs_noisy=20, lr_noisy=1e-4, batch_size=256,
                 proximity_lambda=0.0, dropout_p=0.0):
    preds_train, preds_test, ce_list = [], [], []

    for r in range(R):
        # Noisy fine-tuning always done WITHOUT dropout (eval mode enforced
        # inside train_noisy_model), but architecture must match base_state_dict
        model = make_mlp(cfg["hidden_dim"], cfg["num_layers"],
                         dropout_p=dropout_p)
        model.load_state_dict(base_state_dict)
        # get logit scale preds of training data
        if ADAPTIVE_NOISE_SCALING:
            logit_scale = f_star_train.abs().mean().item()
        else:
            logit_scale = 1.0

        eps = torch.randn_like(f_star_train) * noise_std * logit_scale
        ok = train_noisy_model(
            model, X_train, f_star_train + eps,
            max_epochs=max_epochs_noisy, lr=lr_noisy,
            batch_size=batch_size, ref_state_dict=base_state_dict,
            proximity_lambda=proximity_lambda)
        if not ok:
            print(f"  [WARNING] run {r} diverged, skipping.")
            continue
        model.eval()
        with torch.no_grad():
            preds_train.append(model(X_train).cpu().numpy())
            preds_test.append(model(X_test).cpu().numpy())
            ce_list.append(eval_ce(model, X_test, y_test))

    preds_train = np.stack(preds_train)
    preds_test  = np.stack(preds_test)
    sq_tr = np.sum((preds_train - f_star_train.cpu().numpy()[None]) ** 2, axis=-1)
    sq_te = np.sum((preds_test  - f_star_test.cpu().numpy()[None])  ** 2, axis=-1)
    tpv_tr, tpv_te = float(np.mean(sq_tr)), float(np.mean(sq_te))
    return dict(tpv_train=tpv_tr, tpv_test=tpv_te,
                ce_test_noisy_mean=float(np.mean(ce_list)),
                ce_test_noisy_std=float(np.std(ce_list)))


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
fname = f"results/tpv_mlp_groups_cifar10_{REG_TYPE}_ADAPTIVE_NOISE_SCALING-{ADAPTIVE_NOISE_SCALING}.pkl"


# ----------------------------
# Main loop
# ----------------------------
all_results = []

for mi, cfg in enumerate(model_configs):
    # skip if cfg in all_results already (allows resuming if interrupted)
    if any(r["group_label"] == cfg["group_label"] and r["role"] == cfg["role"]
           for r in all_results):
        print(f"\n=== [{mi+1}/{n_models}] {cfg['group_label']} — {cfg['role']} "
              f"(already done, skipping) ===")
        continue
    print(f"\n=== [{mi+1}/{n_models}] {cfg['group_label']} — {cfg['role']} "
          f"(h={cfg['hidden_dim']}, {REG_TYPE}={cfg['reg_level']:.0e}, "
          f"p={cfg['num_params']:,}) ===")

    # For dropout: bake dropout_p into the architecture itself
    dropout_p = cfg["reg_level"] if REG_TYPE == "dropout" else 0.0
    model_clean = make_mlp(cfg["hidden_dim"], cfg["num_layers"],
                           dropout_p=dropout_p)

    train_clean_model(
        model_clean, X_train, y_train, X_test, y_test,
        max_epochs=max_epochs_clean, lr=lr_clean,
        batch_size=batch_size, momentum=0.9,
        reg_type=REG_TYPE, reg_level=cfg["reg_level"],
    )

    # Always evaluate CE without label smoothing / dropout for clean comparison
    model_clean.eval()
    ce_train_ref = eval_ce(model_clean, X_train, y_train)
    ce_test_ref  = eval_ce(model_clean, X_test,  y_test)
    print(f"  Ref CE — train: {ce_train_ref:.4f}  test: {ce_test_ref:.4f}")

    with torch.no_grad():
        f_star_train = model_clean(X_train)
        f_star_test  = model_clean(X_test)
    base_state_dict = copy.deepcopy(model_clean.state_dict())

    # For TPV estimation, re-instantiate with same dropout_p so architecture matches
    tpv = estimate_tpv(
        cfg=cfg, noise_std=noise_std, R=R,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        base_state_dict=base_state_dict,
        f_star_train=f_star_train, f_star_test=f_star_test,
        max_epochs_noisy=max_epochs_noisy, lr_noisy=lr_noisy,
        batch_size=batch_size, proximity_lambda=proximity_lambda,
        dropout_p=dropout_p,
    )

    all_results.append(dict(**cfg, ce_train_ref=ce_train_ref,
                            ce_test_ref=ce_test_ref, **tpv))
    print(f"  TPV (test): {tpv['tpv_test']:.4e}")

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

# One marker per regularisation level (least → most regularised)
all_markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
marker_map  = {f"level_{i}": all_markers[i] for i in range(N_LEVELS)}

# Human-readable level labels for the legend
reg_unit = {"wd": "wd", "dropout": "p_drop", "labelsmooth": "labelsmooth"}[REG_TYPE]
level_labels = {
    f"level_{i}": f"{reg_unit}={lv} ({'less' if i==0 else 'more'} reg.)"
    if i in (0, N_LEVELS-1)
    else f"{reg_unit}={lv}"
    for i, lv in enumerate(LEVELS)
}

tpv     = np.array([r["tpv_test"]     for r in results])
ce_test = np.array([r["ce_test_ref"]  for r in results])

rho, _ = stats.spearmanr(tpv, ce_test)
print(f"\nSpearman(TPV, CE test) = {rho:.3f}")

LABEL_FONTSIZE  = 20
TICK_FONTSIZE   = 20
TITLE_FONTSIZE  = 15
LEGEND_FONTSIZE = 14

# -------------------------------------------------------
# Figure 1 (KEY FIGURE): two-panel scatter
# -------------------------------------------------------
fig, axes = plt.subplots(1, 1, figsize=(9, 5))

ax2 = axes.twinx()
for r in results:
    xval = r["tpv_test"]
    axes.scatter(xval, r["ce_test_ref"],
               color=color_map[r["group_label"]],
               marker=marker_map[r["role"]],
               s=120, zorder=3, linewidths=0.5, edgecolors="k")
    ax2.scatter(xval, r["ce_train_ref"],
                color=color_map[r["group_label"]],
                marker=marker_map[r["role"]],
                s=50, zorder=3, linewidths=0.2, edgecolors="k", alpha=0.4)

# Connect same-color (same-model size) markers with dashed lines, sorted by TPV
for gl in group_labels:
    sub = [r for r in results if r["group_label"] == gl]
    sub.sort(key=lambda r: r["tpv_test"])
    xs_line = [r["tpv_test"] for r in sub]
    ys_line = [r["ce_test_ref"] for r in sub]
    axes.plot(xs_line, ys_line, "--", color=color_map[gl],
              linewidth=1.2, alpha=0.7, zorder=2)

axes.set_xscale("log")
log_x = np.log10(tpv)
coef  = np.polyfit(log_x, ce_test, 1)
xs    = np.logspace(log_x.min(), log_x.max(), 200)
axes.set_xlabel("TPV (test)", fontsize=LABEL_FONTSIZE)
axes.set_ylabel("Test CE Loss (solid markers)", fontsize=LABEL_FONTSIZE)
ax2.set_ylabel("Train CE Loss (faded markers)", fontsize=LABEL_FONTSIZE)
axes.tick_params(labelsize=TICK_FONTSIZE)
ax2.tick_params(labelsize=TICK_FONTSIZE)

# Legend: groups (colour) + regularisation levels (marker)
legend_elements = (
    [Line2D([0], [0], color=color_map[gl], lw=0, marker="o",
             markersize=8, label=gl) for gl in group_labels]
    + [Line2D([0], [0], color="k", lw=0, marker=marker_map[f"level_{i}"],
               markersize=8, label=level_labels[f"level_{i}"])
       for i in range(N_LEVELS)]
)

axes.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE,
               loc="upper left", framealpha=0, bbox_to_anchor=(0, 0.95))
plt.tight_layout()
plot_path = f"results/plots/mlp_groups_tpv_{REG_TYPE}_ADAPTIVE_NOISE_SCALING-{ADAPTIVE_NOISE_SCALING}.pdf"
plt.savefig(plot_path, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")


# -------------------------------------------------------
# Summary table
# -------------------------------------------------------
print(f"\n{'Model':<30} {'p':>8} {'CE test':>9} {'TPV':>13}")
print("-" * 63)
for r in results:
    label = f"{r['group_label']} {r['role']} ({REG_TYPE}={r['reg_level']:.0e})"
    print(f"{label:<30} {r['num_params']:>8,} {r['ce_test_ref']:>9.4f} "
          f"{r['tpv_test']:>13.4e}")
print(f"\nSpearman(TPV, CE test) = {rho:.3f}")
