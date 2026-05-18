import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Experiment: Sharpness vs Training-set TPV for predicting label-noise sensitivity.

Core claim:
    Sharpness = Tr(H_eff)  does not positively correlate with test-set label-noise sensitivity.
    Training-set TPV          DOES positively correlate with test-set label-noise sensitivity.

Sharpness is estimated as Tr(H_eff) via the doubly-stochastic Hutchinson estimator,
where H_eff = E_x[J(x)^T J(x)] and J(x) = ∂f(x; w)/∂w ∈ R^{K×p} is the
output-parameter Jacobian. This is a label-free quantity:
    Tr(H_eff) ≈ (1/K) Σ_k E_x[ ||J(x)^T v_k||² ]
where v_k ~ {±1}^K are Rademacher random vectors and the expectation over x is
approximated by averaging over the training inputs.

Design:
    - Fix architecture (WIDTH, DEPTH), dataset, test distribution.
    - Train models with the SAME architecture but DIFFERENT weight decay values.
    - For each model compute:
        (a) Sharpness = Tr(H_eff)         via Hutchinson [training data only]
        (b) Training-set TPV              via MC label noise [training data only]
        (c) Ground truth: test-set TPV    via MC label noise [test data]
    - Show:
        * Spearman(sharpness,   TPV_test) is weak or negative
        * Spearman(TPV_train,   TPV_test) is strong and positive
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import stats

from tpv.sgd_noise import SgdTPV
from tpv.label_noise import LabelTPV

# ----------------------------
# Config
# ----------------------------
d         = 20
n_train   = 3000
n_test    = 5000
sigma     = 0.1     # label noise std
R         = 50       # MC runs for empirical TPV
N_HUTCHINSON = 100   # number of Rademacher vectors for Hutchinson estimator
n_epochs_clean  = 100
max_epochs_noisy = 20
lr_clean  = 3e-3
lr_noisy  = 3e-3
WIDTH     = 256
DEPTH     = 4

WD_LIST = [1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 1e-1]

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/plots", exist_ok=True)

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------
# MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, width, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def model_factory():
    return MLP(d, WIDTH, depth=DEPTH)

# ----------------------------
# Training
# ----------------------------
def train_full_batch(model, X, y, max_epochs=1000, train_loss_thres=0.0,
                     lr=1e-2, wd=0.0, print_stats=False,
                     X_test=None, y_test=None):
    model.eval()
    optimizer = optim.AdamW(model.parameters(), lr=lr, #momentum=0.9,
                          weight_decay=wd)
    criterion = nn.MSELoss()
    current_train_loss = float('inf')
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        if not torch.isfinite(loss):
            print(f"  Non-finite loss at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            current_train_loss = criterion(model(X), y).item()
        if current_train_loss < train_loss_thres:
            break
    if print_stats:
        print(f"  Train loss: {current_train_loss:.6f} after {epoch+1} epochs")
    if X_test is not None and y_test is not None:
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
        return test_loss, epoch + 1
    return None, epoch + 1

def train_fn(m, X, y_noisy):
    """Fine-tune m to regress logits to y_noisy via full-batch SGD with MSE."""
    max_epochs_noisy, lr_noisy
    m.eval()  # keep BN/Dropout frozen; only weights change
    optimizer = optim.SGD(m.parameters(), lr=lr_noisy, momentum=0.9, weight_decay=0)
    criterion = nn.MSELoss()
    
    for _ in range(max_epochs_noisy):
        optimizer.zero_grad()
        criterion(m(X), y_noisy).backward()
        optimizer.step()


# ----------------------------
# Data — fixed for all models
# ----------------------------
X_train = torch.randn(n_train, d, device=device)
X_test  = torch.randn(n_test,  d, device=device)
w_true  = torch.randn(d, 1, device=device)
y_clean_train = X_train @ w_true
y_clean_test  = X_test  @ w_true

# ----------------------------
# Main loop
# ----------------------------
n_models = len(WD_LIST)
results = {
    "wd_list":          WD_LIST,
    "p":                None,
    "sigma":            sigma,
    "n_hutchinson":     N_HUTCHINSON,
    "sharpness":        [],    # Tr(H_eff) via Hutchinson — label-free, training data only
    "tpv_train":        [],    # empirical TPV on training set — training data only
    "tpv_test":         [],    # empirical TPV on test set    — ground truth
    "test_loss_clean":  [],
    "train_loss_clean": [],
}

sgd_tpv = SgdTPV(device=device, seed=SEED)


for wi, wd in enumerate(WD_LIST):
    print(f"\n=== WD={wd} ({wi+1}/{n_models}) ===")

    # 1) Train clean reference model w*
    model_clean = MLP(d, WIDTH, depth=DEPTH).to(device)
    test_loss_clean, _ = train_full_batch(
        model_clean, X_train, y_clean_train,
        max_epochs=n_epochs_clean, train_loss_thres=0.0,
        lr=lr_clean, wd=wd, print_stats=True,
        X_test=X_test, y_test=y_clean_test,
    )
    model_clean.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        train_loss_clean = criterion(model_clean(X_train), y_clean_train).item()

    # init_state = {k: v.clone() for k, v in model_clean.state_dict().items()}
    init_state = copy.deepcopy(model_clean.state_dict())

    p = sum(par.numel() for par in model_clean.parameters())
    if results["p"] is None:
        results["p"] = p
    print(f"  Clean test loss: {test_loss_clean:.6f} | "
          f"train loss: {train_loss_clean:.6f} | p={p:,}")

    # 2) Sharpness = Tr(H_eff) via doubly-stochastic Hutchinson estimator (label-free)
    print(f"  Estimating sharpness Tr(H_eff) via Hutchinson ({N_HUTCHINSON} vectors)...")
    _, sharpness = sgd_tpv.compute_tpv(
                model_clean, X_train, y_clean_train,
                lr=0.1, # not being used
                batch_size=1,  # not being used
                n_hutchinson_samples=N_HUTCHINSON,
            )
    print(f"  Tr(H_eff) = {sharpness:.4f}")

    # 3) Training-set TPV (training data only — practitioner diagnostic)
    print(f"  Estimating training-set TPV and test-set TPV ({R} runs)...")
    label_tpv = LabelTPV(device=device, seed=SEED)
    result = label_tpv.compute_tpv(
        model_factory=model_factory,
        base_state_dict=init_state,
        X_train=X_train,
        X_test=X_test,
        noise_std=sigma,
        R=R,
        train_fn=train_fn,
    )

    tpv_train = result['empirical_TPV_train']
    tpv_test = result['empirical_TPV_test']

    print(f"  TPV_train = {tpv_train:.6e}")
    print(f"  TPV_test  = {tpv_test:.6e}")

    results["sharpness"].append(sharpness)
    results["tpv_train"].append(tpv_train)
    results["tpv_test"].append(tpv_test)
    results["test_loss_clean"].append(test_loss_clean)
    results["train_loss_clean"].append(train_loss_clean)

# Convert to arrays
for k in ["sharpness", "tpv_train", "tpv_test",
          "test_loss_clean", "train_loss_clean"]:
    results[k] = np.array(results[k])

# Save
save_path = f"{save_dir}/results.pkl"
with open(save_path, "wb") as f:
    pkl.dump(results, f)
print(f"\nSaved to {save_path}")

# ----------------------------
# Analysis: Spearman correlations
# ----------------------------
sharpness = results["sharpness"]
tpv_train = results["tpv_train"]
tpv_test  = results["tpv_test"]

r_sharp, p_sharp = stats.spearmanr(sharpness, tpv_test)
r_train, p_train = stats.spearmanr(tpv_train, tpv_test)

print(f"\n=== Spearman correlations with test-set TPV (ground truth) ===")
print(f"  Tr(H_eff) vs TPV_test : {r_sharp:.3f}  (p={p_sharp:.4f})")
print(f"  TPV_train vs TPV_test : {r_train:.3f}  (p={p_train:.4f})")

print(f"\n=== Values per model ===")
print(f"{'WD':<10} {'Sharpness':>12} {'TPV_train':>14} {'TPV_test':>14} "
      f"{'Clean test loss':>16}")
for i, wd in enumerate(WD_LIST):
    print(f"  {wd:<8} {sharpness[i]:>12.4f} {tpv_train[i]:>14.6e} "
          f"{tpv_test[i]:>14.6e} {results['test_loss_clean'][i]:>16.6f}")

# ----------------------------
# Plotting
# ----------------------------
LABEL_FONTSIZE  = 22
TICK_FONTSIZE   = 8
TITLE_FONTSIZE  = 16
LEGEND_FONTSIZE = 18

colors = plt.cm.viridis(np.linspace(0, 1, n_models))

# ---- Figure: Sharpness vs TPV_test  |  Label noise TPV_train vs TPV_test ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i, wd in enumerate(WD_LIST):
    ax.scatter(sharpness[i], tpv_test[i], color=colors[i], s=120, zorder=3)
if np.unique(sharpness).size > 1:
    coef = np.polyfit(sharpness, tpv_test, 1)
    xs = np.linspace(sharpness.min(), sharpness.max(), 100)
    ax.plot(xs, np.polyval(coef, xs), "--", color="gray", linewidth=1.5)
ax.set_xlabel("Sharpness  Tr($H_{\\mathrm{eff}}$)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Test-set TPV  (ground truth)", fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)

ax = axes[1]
for i, wd in enumerate(WD_LIST):
    ax.scatter(tpv_train[i], tpv_test[i], color=colors[i], s=120, zorder=3)
if np.unique(tpv_train).size > 1:
    coef = np.polyfit(tpv_train, tpv_test, 1)
    xs = np.linspace(tpv_train.min(), tpv_train.max(), 100)
    ax.plot(xs, np.polyval(coef, xs), "--", color="gray", linewidth=1.5)
mn = min(tpv_train.min(), tpv_test.min())
mx = max(tpv_train.max(), tpv_test.max())
# ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.4,
#         label="y=x (reference line)")
ax.set_xlabel("Training-set TPV  (no test labels)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Test-set TPV  (ground truth)", fontsize=LABEL_FONTSIZE)

ax.tick_params(labelsize=TICK_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
plt.savefig(f"{save_dir}/plots/sharpness_vs_tpv_train_vs_tpv_test.pdf",
            bbox_inches="tight")

plt.show()
print(f"\nDone. Plots saved to {save_dir}/plots/")
