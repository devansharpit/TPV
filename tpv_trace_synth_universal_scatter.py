"""

Synthetic experiments for TPV trace stability:
- Multiple data generation processes
- Architecture variations (input_dim, width, depth)
- Two perturbation sources:
    1) Label noise
    2) SGD stationary noise near convergence
- Variation in number of training samples (n_train)

For each setting, we compute empirical TPV on train and test:
    TPV_train  = E_{runs,x in train} (f_r(x) - f_star(x))^2
    TPV_test   = E_{runs,x in test}  (f_r(x) - f_star(x))^2

We also compute train/test MSE for the noisy models (label-noise
or SGD-noise) and use L_test - L_train as the generalization gap.

Results are saved to a pickle file and the script is resumable:
already-completed settings are skipped.

The plotting function makes a *unified* scatter plot where:
    x-axis: TPV_train
    y-axis: TPV_test
    color:  generalization gap (L_test - L_train)
    marker: number of training samples (n_train = 10 vs 1000)
"""

import os
import pickle as pkl
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# ----------------------------
# Global config
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = f"{RESULTS_DIR}/tpv_synthetic_results.pkl" 
RNG_SEED = 0

# Input dims / dataset sizes
INPUT_DIM_LIST = [10, 20, 50]   # variation over input dimensionality
N_TRAIN_LIST = [10, 1000]       # number of training samples (small vs large)
N_TEST = 5000                   # keep test size fixed and large

# Number of samples used to TRAIN the clean reference model f*.
# This is independent of the n_train used when evaluating TPV.
REF_N_TRAIN_FOR_CLEAN = 1000  # e.g., 1000

# Architecture sweeps
WIDTH_LIST = [1, 256, ]
DEPTH_LIST = [2, 3, 4]          # number of layers in the MLP (>= 2)

# Data generation modes
DATASET_TYPES = [
    "multi_relu_teacher",       # X ~ N(0, I), y = sum_k ReLU(a_k^T x + b_k)
    "linear_gaussian",          # X ~ N(0, I), y = x^T w_true
    "relu_teacher",             # X ~ N(0, I), y = ReLU(a^T x)
    # "two_gaussian_mixture",     # mixture of two Gaussians + linear teacher
]

# Label-noise TPV sweeps
LABEL_SIGMA_LIST = [0.005, 0.01] # [0.05, 0.1]  # std of additive label noise
LABEL_TPV_RUNS = 20             # Monte Carlo runs per (config, sigma)

# SGD-noise TPV sweeps
SGD_LR_LIST = [1e-3, 5e-4]      # learning rates controlling SGD noise strength
SGD_BATCH_SIZE_LIST = [32, 128]
SGD_STEPS = 1000                # total SGD steps for sampling around the minimum
SGD_BURN_IN = 200               # number of initial steps to discard
SGD_SNAPSHOT_EVERY = 20         # take a TPV sample every K steps after burn-in

# Training hyperparameters for clean model and label-noise runs
CLEAN_EPOCHS = 800
# CLEAN_LR = 5e-3
CLEAN_LR = 2e-3

LABEL_NOISE_EPOCHS = 200
# LABEL_NOISE_LR = 5e-3
LABEL_NOISE_LR = 2e-3

# How strongly to penalize deviation from the clean reference parameters w*
# during label-noise training. Set to 0.0 to disable. Perturbed parameter w 
# should not be too far from w* in order for the first order approximation 
# to hold in the TPV approximation.
CENTER_PENALTY_LAMBDA = 1e-3 # 1e-0

# ----------------------------
# First-order Taylor rejection test
# ----------------------------
# Number of reference points to test the Taylor approximation on
TAYLOR_N_REF = 128   # number of reference inputs for the Taylor test
TAYLOR_H = 1e-2      # finite-difference step size along (w - w*)


# ----------------------------
# Utils: reproducibility
# ----------------------------

def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ----------------------------
# Helper: proximity penalty
# ----------------------------
def compute_proximity_penalty(model, ref_state_dict):
    """
    Compute L2 proximity penalty: sum of ||w - w_ref||^2 over all parameters.
    """
    penalty = 0.0
    num_params = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        if name in ref_state_dict:
            ref_param = ref_state_dict[name]
            penalty = penalty + torch.sum((param - ref_param) ** 2)
    return penalty 

# ----------------------------
# First order Taylor check
# ----------------------------

def compute_taylor_rel_err(model_run,
                           input_dim,
                           width,
                           depth,
                           base_state_dict,
                           X_ref,
                           f_star_ref,
                           h=TAYLOR_H,
                           device=DEVICE):
    """Compute relative first-order Taylor error of model_run around w*.

    We test along the displacement Δ = w - w* using finite differences:
        d(x) = f_w(x) - f_{w*}(x)
        g(x) = [f(w* + h Δ; x) - f_{w*}(x)] / h

    and return
        rel_err = E[(d-g)^2] / (E[d^2] + eps).

    This is a scalar summary of how well a first-order Taylor expansion
    around w* describes the change from w* to w on the reference set X_ref.
    """
    X_ref = X_ref.to(device)
    f_star_ref = f_star_ref.to(device)

    model_run = model_run.to(device)
    model_run.eval()
    with torch.no_grad():
        f_run_ref = model_run(X_ref)

    # Build finite-difference model at w* + h (w - w*)
    model_fd = MLP(input_dim, width, depth=depth).to(device)

    state_star = {k: v.to(device) for k, v in base_state_dict.items()}
    state_run = model_run.state_dict()

    state_fd = {}
    for k in state_star.keys():
        v_star = state_star[k]
        v_run = state_run[k]
        delta = v_run - v_star
        state_fd[k] = v_star + h * delta

    model_fd.load_state_dict(state_fd)
    model_fd.eval()
    with torch.no_grad():
        f_fd_ref = model_fd(X_ref)

    d = f_run_ref - f_star_ref
    g = (f_fd_ref - f_star_ref) / h

    num = ((d - g) ** 2).mean().item()
    denom = (d ** 2).mean().item() + 1e-12
    rel_err = num / denom
    return rel_err


# ----------------------------
# Model definition
# ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int = 3):
        super().__init__()
        layers = []
        if depth < 2:
            raise ValueError("Depth must be >= 2 (input->hidden(s)->output).")

        # First hidden layer
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())

        # Middle hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer (scalar regression)
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_full_batch(model,
                     X,
                     y,
                     n_epochs=500,
                     lr=1e-2,
                     wd=0.0,
                     center_params=None,
                     center_lambda=0.0,
                     verbose=False):
    # model.train()
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    preds = model(X)
    initial_loss = criterion(preds, y)
    print(f"  Initial training loss: {initial_loss.item():.6f}")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)


        # Quadratic penalty around w* : lambda * ||w - w*||^2
        if center_params is not None and center_lambda > 0.0:
            penalty = compute_proximity_penalty(model, center_params)
            loss = loss + center_lambda * penalty

        # Early abort if things go off the rails
        if not torch.isfinite(loss):
            print(f"Non-finite loss at epoch {epoch}: {loss.item()}")
            break
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

    # if verbose:
    print(f"  Final training loss: {loss.item():.6f}")
    if loss>initial_loss:
        print("Warning: Training loss increased during training.")

    # NaN check
    for p in model.parameters():
        if torch.isnan(p).any():
            print("\n[WARNING]: NaN detected in model parameters after training.\n")
    return initial_loss, loss


# ----------------------------
# Data generation
# ----------------------------

def generate_dataset(dataset_type: str,
                     n_train: int,
                     n_test: int,
                     d: int,
                     device=DEVICE):
    """
    Returns:
        X_train, y_train, X_test, y_test, meta
    meta is a dict with info about the underlying teacher.
    """
    if dataset_type == "linear_gaussian":
        # X ~ N(0, I), y = x^T w_true
        X_train = torch.randn(n_train, d, device=device)
        X_test = torch.randn(n_test, d, device=device)
        w_true = torch.randn(d, 1, device=device)
        y_train = X_train @ w_true
        y_test = X_test @ w_true
        meta = {"w_true": w_true.detach().cpu().numpy()}

    elif dataset_type == "relu_teacher":
        # X ~ N(0, I), y = ReLU(a^T x)
        X_train = torch.randn(n_train, d, device=device)
        X_test = torch.randn(n_test, d, device=device)
        a = torch.randn(d, 1, device=device)
        y_train = torch.relu(X_train @ a)
        y_test = torch.relu(X_test @ a)
        meta = {"a": a.detach().cpu().numpy()}

    elif dataset_type == "two_gaussian_mixture":
        # Mixture of two Gaussians with different means; linear teacher
        mean1 = torch.ones(d, device=device)
        mean2 = -torch.ones(d, device=device)

        def sample_mixture(n):
            comp = torch.randint(low=0, high=2, size=(n,), device=device)
            noise = torch.randn(n, d, device=device)
            X = torch.where(comp.unsqueeze(1) == 0,
                            noise + mean1,
                            noise + mean2)
            return X

        X_train = sample_mixture(n_train)
        X_test = sample_mixture(n_test)
        w_true = torch.randn(d, 1, device=device)
        y_train = X_train @ w_true
        y_test = X_test @ w_true
        meta = {
            "w_true": w_true.detach().cpu().numpy(),
            "mean1": mean1.detach().cpu().numpy(),
            "mean2": mean2.detach().cpu().numpy(),
        }

    elif dataset_type == "multi_relu_teacher":
        # X ~ N(0, I), y = sum_{k=1}^K ReLU(a_k^T x + b_k)
        X_train = torch.randn(n_train, d, device=device)
        X_test  = torch.randn(n_test,  d, device=device)

        K = 10  # number of hidden units in teacher

        A = torch.randn(d, K, device=device)          # weights a_k
        b = 0.5 * torch.randn(K, device=device)       # biases b_k

        def f(X):
            # X: (n, d)
            pre = X @ A + b   # (n, K)
            return torch.relu(pre).sum(dim=1, keepdim=True)  # (n, 1)

        y_train = f(X_train)
        y_test  = f(X_test)

        meta = {
            "A": A.detach().cpu().numpy(),
            "b": b.detach().cpu().numpy(),
        }

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return X_train, y_train, X_test, y_test, meta


# ----------------------------
# Label-noise TPV experiment
# ----------------------------

def estimate_tpv_label_noise(input_dim,
                             width,
                             depth,
                             sigma,
                             R,
                             X_train,
                             y_clean_train,
                             X_test,
                             y_clean_test,
                             n_epochs,
                             lr,
                             init_state,
                             f_star_train,
                             f_star_test,
                             device=DEVICE):
    """Label-noise TPV with per-run statistics and Taylor errors.

    For each of R runs:
      - Add Gaussian label noise N(0, sigma^2) to y_clean_train.
      - Retrain from the same initialization init_state.
      - Measure prediction fluctuations around the clean reference f_star
        on train and test.
      - Compute a scalar TPV contribution on train/test, per-run MSE, and
        a first-order Taylor relative error around w* (init_state).
    """
    f_star_train = f_star_train.to(device)
    f_star_test = f_star_test.to(device)

    tpv_train_per_run = []
    tpv_test_per_run = []
    L_train_per_run = []
    L_test_per_run = []
    taylor_rel_err_per_run = []
    init_final_loss_per_run = []

    # Build a small reference set for the Taylor test (shared across runs)
    n_ref = min(TAYLOR_N_REF, X_train.shape[0])
    X_ref = X_train[:n_ref].detach()
    with torch.no_grad():
        base_model = MLP(input_dim, width, depth=depth).to(device)
        base_model.load_state_dict(init_state)
        base_model.eval()
        f_star_ref = base_model(X_ref).detach()

    for r in range(R):
        model = MLP(input_dim, width, depth=depth).to(device)
        model.load_state_dict(init_state)

        # Add label noise
        eps = torch.randn_like(y_clean_train) * sigma
        y_noisy = y_clean_train + eps

        initial_loss, final_loss = train_full_batch(
            model,
            X_train,
            y_noisy,
            n_epochs=n_epochs,
            lr=lr,
            wd=0.0,
            center_lambda=CENTER_PENALTY_LAMBDA,
            verbose=False,
        )

        # First-order Taylor relative error for this run
        rel_err = compute_taylor_rel_err(
            model_run=model,
            input_dim=input_dim,
            width=width,
            depth=depth,
            base_state_dict=init_state,
            X_ref=X_ref,
            f_star_ref=f_star_ref,
            h=TAYLOR_H,
            device=device,
        )
        taylor_rel_err_per_run.append(float(rel_err))

        model.eval()
        with torch.no_grad():
            train_preds = model(X_train)   # (n_train, 1)
            test_preds = model(X_test)     # (n_test, 1)

        # Per-run TPV contributions (mean squared deviation from f_star)
        diff_train = train_preds - f_star_train
        diff_test = test_preds - f_star_test
        tpv_train_inst = float(diff_train.pow(2).mean().item())
        tpv_test_inst = float(diff_test.pow(2).mean().item())

        # Per-run MSE w.r.t. clean targets
        mse_train_inst = float((train_preds - y_clean_train).pow(2).mean().item())
        mse_test_inst = float((test_preds - y_clean_test).pow(2).mean().item())

        tpv_train_per_run.append(tpv_train_inst)
        tpv_test_per_run.append(tpv_test_inst)
        L_train_per_run.append(mse_train_inst)
        L_test_per_run.append(mse_test_inst)
        init_final_loss_per_run.append((float(initial_loss.item()), float(final_loss.item())))

    # Aggregate over all runs (no Taylor-based rejection here)
    tpv_train_mean = float(np.mean(tpv_train_per_run))
    tpv_test_mean = float(np.mean(tpv_test_per_run))
    L_train_mean = float(np.mean(L_train_per_run))
    L_test_mean = float(np.mean(L_test_per_run))

    return (tpv_train_mean,
            tpv_test_mean,
            L_train_mean,
            L_test_mean,
            tpv_train_per_run,
            tpv_test_per_run,
            L_train_per_run,
            L_test_per_run,
            init_final_loss_per_run,
            taylor_rel_err_per_run)




def compute_tpv_from_runs(preds_runs_train,
                          preds_runs_test,
                          f_star_train,
                          f_star_test):
    """
    Given:
        preds_runs_train: (R, n_train)
        preds_runs_test:  (R, n_test)
        f_star_*: torch tensors on device or CPU, shape (n, 1)
    Returns:
        empirical_TPV_train, empirical_TPV_test
    """
    f_star_train_np = f_star_train.squeeze(-1).cpu().numpy()
    f_star_test_np = f_star_test.squeeze(-1).cpu().numpy()

    diffs_train = preds_runs_train - f_star_train_np[None, :]
    diffs_test = preds_runs_test - f_star_test_np[None, :]

    tpv_train = float(np.mean(diffs_train ** 2))
    tpv_test = float(np.mean(diffs_test ** 2))
    return tpv_train, tpv_test


# ----------------------------
# SGD-noise TPV experiment
# ----------------------------

def build_data_loader(X, y, batch_size):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def estimate_tpv_sgd_noise(input_dim,
                           width,
                           depth,
                           sgd_lr,
                           batch_size,
                           X_train,
                           y_clean_train,
                           X_test,
                           y_clean_test,
                           base_model_state,
                           sgd_steps=SGD_STEPS,
                           burn_in=SGD_BURN_IN,
                           snapshot_every=SGD_SNAPSHOT_EVERY,
                           device=DEVICE):
    """SGD-noise TPV with per-snapshot statistics and Taylor errors."""
    # Build a model and load clean solution
    model = MLP(input_dim, width, depth=depth).to(device)
    model.load_state_dict(base_model_state)

    # Compute f_star on train/test once
    model.eval()
    with torch.no_grad():
        f_star_train = model(X_train)
        f_star_test = model(X_test)

    # Build reference set for Taylor test
    n_ref = min(TAYLOR_N_REF, X_train.shape[0])
    X_ref = X_train[:n_ref].detach()
    with torch.no_grad():
        f_star_ref = model(X_ref).detach()

    # Prepare SGD around this solution
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=sgd_lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.MSELoss()
    loader = build_data_loader(X_train, y_clean_train, batch_size)

    step = 0
    snapshot_count = 0
    tpv_train_per_snap = []
    tpv_test_per_snap = []
    L_train_per_snap = []
    L_test_per_snap = []
    taylor_rel_err_per_snap = []

    data_iter = iter(loader)

    center_params = copy.deepcopy(model.state_dict())
    
    num_params = sum(p.numel() for p in model.parameters())
    while step < sgd_steps:
        try:
            X_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            X_batch, y_batch = next(data_iter)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        if center_params is not None and CENTER_PENALTY_LAMBDA > 0.0:
            penalty = compute_proximity_penalty(model, center_params)
            loss = loss + CENTER_PENALTY_LAMBDA * penalty

        loss.backward()
        optimizer.step()

        step += 1

        # Sample model predictions after burn-in, every snapshot_every steps
        if step >= burn_in and (step - burn_in) % snapshot_every == 0:
            snapshot_count += 1

            # Taylor relative error for this snapshot
            rel_err = compute_taylor_rel_err(
                model_run=model,
                input_dim=input_dim,
                width=width,
                depth=depth,
                base_state_dict=base_model_state,
                X_ref=X_ref,
                f_star_ref=f_star_ref,
                h=TAYLOR_H,
                device=device,
            )
            taylor_rel_err_per_snap.append(float(rel_err))

            model.eval()
            with torch.no_grad():
                train_preds = model(X_train)
                test_preds = model(X_test)

                diff_train = (train_preds - f_star_train)
                diff_test = (test_preds - f_star_test)

                tpv_train_inst = float(diff_train.pow(2).mean().item())
                tpv_test_inst = float(diff_test.pow(2).mean().item())

                mse_train_inst = float(((train_preds - y_clean_train) ** 2).mean().item())
                mse_test_inst = float(((test_preds - y_clean_test) ** 2).mean().item())

            tpv_train_per_snap.append(tpv_train_inst)
            tpv_test_per_snap.append(tpv_test_inst)
            L_train_per_snap.append(mse_train_inst)
            L_test_per_snap.append(mse_test_inst)

            # model.train()
            model.eval()

    if snapshot_count == 0:
        raise RuntimeError("No snapshots collected; adjust burn_in / snapshot_every / sgd_steps.")

    tpv_train_mean = float(np.mean(tpv_train_per_snap))
    tpv_test_mean = float(np.mean(tpv_test_per_snap))
    L_train_mean = float(np.mean(L_train_per_snap))
    L_test_mean = float(np.mean(L_test_per_snap))

    return (tpv_train_mean,
            tpv_test_mean,
            L_train_mean,
            L_test_mean,
            tpv_train_per_snap,
            tpv_test_per_snap,
            L_train_per_snap,
            L_test_per_snap,
            taylor_rel_err_per_snap)




# ----------------------------
# Results handling (resumable)
# ----------------------------

def load_results(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            results = pkl.load(f)
        print(f"Loaded existing results from {path} with {len(results)} entries.")
    else:
        results = {}
        print(f"No existing results at {path}; starting fresh.")
    return results


def save_results(results: dict, path: str):
    with open(path, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved results to {path} (entries: {len(results)}).")


# ----------------------------
# Main experiment driver
# ----------------------------
def main():
    set_global_seed(RNG_SEED)

    results = load_results(RESULTS_PATH)

    bad_run_cnt = 0
    bad_run_taylor_thres = 0.001

    # ------------------------------------------------------------------
    # 1) Generate teacher data ONCE per (dataset_type, input_dim),
    #    always with REF_N_TRAIN_FOR_CLEAN training samples.
    #    All n_train values will be subsets of this pool.
    # ------------------------------------------------------------------
    teacher_cache = {}
    for ds_type in DATASET_TYPES:
        for in_dim in INPUT_DIM_LIST:
            X_full, y_full, X_test, y_test, meta = generate_dataset(
                ds_type,
                REF_N_TRAIN_FOR_CLEAN,
                N_TEST,
                in_dim,
                device=DEVICE,
            )
            teacher_cache[(ds_type, in_dim)] = (X_full, y_full, X_test, y_test, meta)

    # ------------------------------------------------------------------
    # 2) Loop over (dataset_type, input_dim, n_train, width, depth)
    #    For each n_train, we take a subset of size n_train from X_full,
    #    but the clean reference model is always trained on X_full.
    # ------------------------------------------------------------------
    for ds_type, input_dim, n_train, width, depth in product(
        DATASET_TYPES, INPUT_DIM_LIST, N_TRAIN_LIST, WIDTH_LIST, DEPTH_LIST
    ):
        print(
            f"\n=== Dataset: {ds_type}, input_dim: {input_dim}, "
            f"n_train (TPV eval): {n_train}, width: {width}, depth: {depth} ==="
        )

        if n_train > REF_N_TRAIN_FOR_CLEAN:
            raise ValueError(
                f"Requested n_train={n_train} > REF_N_TRAIN_FOR_CLEAN="
                f"{REF_N_TRAIN_FOR_CLEAN}. Increase REF_N_TRAIN_FOR_CLEAN or "
                "reduce N_TRAIN_LIST."
            )

        X_full, y_full, X_test, y_test, _ = teacher_cache[(ds_type, input_dim)]

        # Subset used for TPV evaluation and noisy retraining
        X_train = X_full[:n_train]
        y_clean_train = y_full[:n_train]
        y_clean_test = y_test  # same test set regardless of n_train

        # ------------------------------------------------------------------
        # 2a) Train / load clean reference model f* for this
        #     (ds_type, input_dim, width, depth).
        #     It is ALWAYS trained on REF_N_TRAIN_FOR_CLEAN samples.
        # ------------------------------------------------------------------
        clean_key = (
            "clean_reference",
            ds_type,
            input_dim,
            width,
            depth,
            REF_N_TRAIN_FOR_CLEAN,
        )

        if clean_key in results:
            print("  Clean reference (fixed N_ref) already computed; reusing.")
            base_state_dict = results[clean_key]["state_dict"]
            f_star_train_full = results[clean_key]["f_star_train_full"]
            f_star_test = results[clean_key]["f_star_test"]
        else:
            print(
                "  Training clean reference model on "
                f"N_ref = {REF_N_TRAIN_FOR_CLEAN} samples..."
            )
            base_model = MLP(input_dim, width, depth=depth).to(DEVICE)

            # Uses current train_full_batch, which prints initial/final loss
            # and warns if training loss increased.
            train_full_batch(
                base_model,
                X_full,
                y_full,
                n_epochs=CLEAN_EPOCHS,
                lr=CLEAN_LR,
                wd=0.0,
                verbose=True,
            )

            base_model.eval()
            with torch.no_grad():
                f_star_train_full = base_model(X_full).detach().cpu()
                f_star_test = base_model(X_test).detach().cpu()

                # Clean MSE on the full reference training pool (for logging/storage)
                y_full_cpu = y_full.detach().cpu()
                y_test_cpu = y_test.detach().cpu()
                L_train_full = float(
                    ((f_star_train_full - y_full_cpu) ** 2).mean().item()
                )
                L_test_full = float(
                    ((f_star_test - y_test_cpu) ** 2).mean().item()
                )

            base_state_dict = base_model.state_dict()

            results[clean_key] = {
                "state_dict": base_state_dict,
                "f_star_train_full": f_star_train_full,
                "f_star_test": f_star_test,
                "L_train_full": L_train_full,
                "L_test_full": L_test_full,
            }
            save_results(results, RESULTS_PATH)

        # Slice f* to match the current n_train subset
        f_star_train = f_star_train_full[:n_train]

        # Compute clean train/test MSE for THIS n_train (for logging only)
        with torch.no_grad():
            y_clean_train_cpu = y_clean_train.detach().cpu()
            y_clean_test_cpu = y_clean_test.detach().cpu()
            L_train_clean = float(
                ((f_star_train - y_clean_train_cpu) ** 2).mean().item()
            )
            L_test_clean = float(
                ((f_star_test - y_clean_test_cpu) ** 2).mean().item()
            )
        print(
            f"  Clean MSE train (n={n_train}): {L_train_clean:.4e}, "
            f"test: {L_test_clean:.4e}"
        )

        # ------------------------------------------------------------------
        # 2b) Label-noise TPV experiments (unchanged interface, but now
        #     using fixed-N reference model and n_train subset).
        # ------------------------------------------------------------------
        for sigma in LABEL_SIGMA_LIST:
            key = ("label_noise", ds_type, input_dim, n_train, width, depth, sigma)
            if key in results:
                print(f"  [Label noise] sigma={sigma} already done; skipping.")
                continue

            print(f"  [Label noise] sigma={sigma}: running {LABEL_TPV_RUNS} runs...")
            (
                tpv_train,
                tpv_test,
                L_train_noisy,
                L_test_noisy,
                tpv_train_per_run,
                tpv_test_per_run,
                L_train_per_run,
                L_test_per_run,
                init_final_loss_per_run,
                taylor_rel_err_per_run,
            ) = estimate_tpv_label_noise(
                input_dim=input_dim,
                width=width,
                depth=depth,
                sigma=sigma,
                R=LABEL_TPV_RUNS,
                X_train=X_train,
                y_clean_train=y_clean_train,
                X_test=X_test,
                y_clean_test=y_clean_test,
                n_epochs=LABEL_NOISE_EPOCHS,
                lr=LABEL_NOISE_LR,
                init_state=base_state_dict,
                f_star_train=f_star_train,
                f_star_test=f_star_test,
                device=DEVICE,
            )

            results[key] = {
                "source": "label_noise",
                "tpv_train": tpv_train,
                "tpv_test": tpv_test,
                "sigma": sigma,
                "R": LABEL_TPV_RUNS,
                "L_train": L_train_noisy,
                "L_test": L_test_noisy,
                "tpv_train_per_run": tpv_train_per_run,
                "tpv_test_per_run": tpv_test_per_run,
                "L_train_per_run": L_train_per_run,
                "L_test_per_run": L_test_per_run,
                "init_final_loss_per_run": init_final_loss_per_run,
                "taylor_rel_err_per_run": taylor_rel_err_per_run,
            }
            print(f"    -> TPV_train={tpv_train:.6e}, TPV_test={tpv_test:.6e}")
            save_results(results, RESULTS_PATH)

            taylor_rel_err_per_run_np = np.array(taylor_rel_err_per_run)
            mask = taylor_rel_err_per_run_np <= bad_run_taylor_thres
            if not np.any(mask):
                bad_run_cnt += 1
            print(f"Total bad runs so far: {bad_run_cnt}")

        # ------------------------------------------------------------------
        # 2c) SGD-noise TPV experiments (unchanged, but X_train/y_clean_train
        #     now come from the n_train subset; base_model_state is fixed-N).
        # ------------------------------------------------------------------
        for sgd_lr, batch_size in product(SGD_LR_LIST, SGD_BATCH_SIZE_LIST):
            key = (
                "sgd_noise",
                ds_type,
                input_dim,
                n_train,
                width,
                depth,
                sgd_lr,
                batch_size,
            )
            if key in results:
                print(
                    f"  [SGD noise] lr={sgd_lr}, batch={batch_size} already done; skipping."
                )
                continue

            print(
                f"  [SGD noise] lr={sgd_lr}, batch={batch_size}: running SGD dynamics..."
            )
            (
                tpv_train,
                tpv_test,
                L_train_sgd,
                L_test_sgd,
                tpv_train_per_snap,
                tpv_test_per_snap,
                L_train_per_snap,
                L_test_per_snap,
                taylor_rel_err_per_snap,
            ) = estimate_tpv_sgd_noise(
                input_dim=input_dim,
                width=width,
                depth=depth,
                sgd_lr=sgd_lr,
                batch_size=batch_size,
                X_train=X_train,
                y_clean_train=y_clean_train,
                X_test=X_test,
                y_clean_test=y_clean_test,
                base_model_state=base_state_dict,
                sgd_steps=SGD_STEPS,
                burn_in=SGD_BURN_IN,
                snapshot_every=SGD_SNAPSHOT_EVERY,
                device=DEVICE,
            )

            results[key] = {
                "source": "sgd_noise",
                "tpv_train": tpv_train,
                "tpv_test": tpv_test,
                "sgd_lr": sgd_lr,
                "batch_size": batch_size,
                "sgd_steps": SGD_STEPS,
                "burn_in": SGD_BURN_IN,
                "snapshot_every": SGD_SNAPSHOT_EVERY,
                "L_train": L_train_sgd,
                "L_test": L_test_sgd,
                "tpv_train_per_run": tpv_train_per_snap,
                "tpv_test_per_run": tpv_test_per_snap,
                "L_train_per_run": L_train_per_snap,
                "L_test_per_run": L_test_per_snap,
                "taylor_rel_err_per_run": taylor_rel_err_per_snap,
            }
            print(f"    -> TPV_train={tpv_train:.6e}, TPV_test={tpv_test:.6e}")
            save_results(results, RESULTS_PATH)

            taylor_rel_err_per_run_np = np.array(taylor_rel_err_per_snap)
            mask = taylor_rel_err_per_run_np <= bad_run_taylor_thres
            if not np.any(mask):
                bad_run_cnt += 1
            print(f"Total bad runs so far: {bad_run_cnt}")

    print("\nAll synthetic experiments complete.")


# ----------------------------
# Unified TPV scatter plotting
# ----------------------------

def plot_unified_tpv_scatter(results_path,
                             use_log_scale=True, exp_name=None, n_train_exc=None,
                             save_name=None, mode='n', TAYLOR_REL_ERR_TOL = 0.2, cmap='viridis'):
    """
    Make a *single* scatter plot:

        x-axis: TPV_train
        y-axis: TPV_test
        color:  generalization gap (L_test - L_train)
        marker: number of training samples (n_train)

    This lets you visually check that TPV_train ≈ TPV_test even
    for points with large generalization gaps, and also see how
    the small-n_train regime deviates from the y=x trend.
    """
    with open(results_path+f'/{exp_name}.pkl', "rb") as f: # tpv_synthetic_results5
        results = pkl.load(f)

    tpv_train_all = []
    tpv_test_all = []
    gen_gap_all = []
    ntrain_all = []
    source_all = []
    width_all = []


    
    for key, val in results.items():
        if not isinstance(key, tuple) or len(key) == 0:
            continue

        tag = key[0]
        if tag not in ["label_noise", "sgd_noise"]:
            continue

        if tag == "label_noise":
            _, ds_type, input_dim, n_train, width, depth, sigma = key
        else:  # "sgd_noise"
            _, ds_type, input_dim, n_train, width, depth, sgd_lr, batch_size = key

        errs = np.array(val["taylor_rel_err_per_run"])
        mask = errs <= TAYLOR_REL_ERR_TOL
        
        REF_N_TRAIN_FOR_CLEAN = 1000 # n_train
        clean_result_key = ("clean_reference", ds_type, input_dim, width, depth, REF_N_TRAIN_FOR_CLEAN) # 
        clean_result = results[clean_result_key]
        L_train_clean = clean_result["L_train_full"]
        L_test_clean = clean_result["L_test_full"]
        
        
        tpv_train_per = np.array(val["tpv_train_per_run"])
        tpv_test_per  = np.array(val["tpv_test_per_run"])
        init_final_loss_per = val.get("init_final_loss_per", [])
        if init_final_loss_per !=[]:
            initial_loss_per = np.array([t[0] for t in init_final_loss_per])
            final_loss_per =  np.array([t[1] for t in init_final_loss_per])
            loss_reduce = final_loss_per<initial_loss_per
            mask = mask & loss_reduce
        
        # Skip configs where all runs are rejected
        if not np.any(mask):
            continue
        
        tpv_train_filtered = tpv_train_per[mask].mean()
        tpv_test_filtered  = tpv_test_per[mask].mean()
        L_train_filtered   = L_train_clean 
        L_test_filtered    = L_test_clean 

        
        # Optional: skip specific n_train
        if (n_train_exc is not None) and (n_train == n_train_exc):
            continue
        
        # Optional critr filter
        if mode == "n":
            critr = (width == 1)
        else:
            critr = (n_train == 10)
            
        if np.isnan(val["tpv_train"]) or np.isnan(val["tpv_test"]) or (n_train_exc is not None and n_train==n_train_exc)\
        or  critr: # or ds_type!='multi_relu_teacher' :# or  width==1:# or tag=='label_noise'
            continue
        tpv_train_all.append(tpv_train_filtered)
        tpv_test_all.append(tpv_test_filtered)
        gen_gap_all.append(L_test_filtered - L_train_filtered)
        ntrain_all.append(n_train)
        width_all.append(width)
        source_all.append(val.get("source", tag))  # robust to older pickles

    tpv_train_all = np.array(tpv_train_all)
    tpv_test_all = np.array(tpv_test_all)
    gen_gap_all = np.array(gen_gap_all)
    ntrain_all = np.array(ntrain_all)
    width_all = np.array(width_all)

    print(f'tpv_train_all shape {tpv_train_all.shape}')
    print(np.unique(ntrain_all))
    print(np.unique(width_all))

    if tpv_train_all.size == 0:
        print("No TPV entries found to plot.")
        return

    plt.figure(figsize=(4, 3))

    # Two n_train values, two markers
    markers_by_n = {
        10: "o",
        1000: "^",
    }
    
    markers_by_width = {
                        1: "o",
                        # 16: "s",
                        # 64: "^",
                        # 128: "D",
                        256: "P",
                    }

    vmin = np.min(gen_gap_all)
    vmax = np.max(gen_gap_all)

    

    # Plot by n_train with different markers
    min_val, max_val = np.inf, -np.inf
    cnt = 0
    print(mode)
    if mode=='n':
        
        for n_train in sorted(np.unique(ntrain_all)):
            mask = (ntrain_all == n_train) & (tpv_train_all>10e-6) & (tpv_test_all>10e-6)

            min_val =  10e-6#min(min_val, tpv_train_all[mask].min(), tpv_test_all[mask].min()) + 1e-15
            max_val =  max(max_val, tpv_train_all[mask].max(), tpv_test_all[mask].max())
        
            cnt += sum(mask) # len(tpv_train_all[mask])
            plt.scatter(
                tpv_train_all[mask],
                tpv_test_all[mask],
                c=gen_gap_all[mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                marker=markers_by_n.get(int(n_train), "o"),
                # marker=markers_by_width.get(int(width), "o"),
                alpha=0.5,
                label=f"n_train={n_train}",
                edgecolors="none",
                s=30,
            )
        plt.title(f"width={np.unique(width_all)[0]}")
    else:
        # for n_train in sorted(np.unique(ntrain_all)):
        #     mask = (ntrain_all == n_train)
        for width in sorted(np.unique(width_all)):
            mask = (width_all == width)

            min_val =  10e-9#min(min_val, tpv_train_all[mask].min(), tpv_test_all[mask].min()) + 1e-15
            max_val =  max(max_val, tpv_train_all[mask].max(), tpv_test_all[mask].max())
            
            cnt += sum(mask) # len(tpv_train_all[mask])
            plt.scatter(
                tpv_train_all[mask],
                tpv_test_all[mask],
                c=gen_gap_all[mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                # marker=markers_by_n.get(int(n_train), "o"),
                marker=markers_by_width.get(int(width), "o"),
                alpha=0.5,
                label=f"width={width}",
                edgecolors="none",
                s=30,
            )
        plt.title(f"n_train={np.unique(ntrain_all)[0]}")

     # Diagonal y=x line limits
    
    print(cnt, len(tpv_train_all))
    # ----- 10% error band -----
    eb = 0.5
    factor = 1.0 + eb
    
    # Safety margin so line extends slightly beyond data
    # print(min_val, max_val)
    margin = 0.05  # 5% in log space
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_span = log_max - log_min
    log_min -= margin * log_span
    log_max += margin * log_span
    
    # 2) Use logspace on a strictly positive range
    x_line = np.logspace(log_min, log_max, 400)
    
    lower = x_line / factor      # y = x / (1 + eb)
    upper = x_line * factor      # y = x * (1 + eb)

    plt.fill_between(
        x_line, lower, upper,
        color="gray", alpha=0.2,# label="10% band"
    )

    # Central reference line
    plt.plot(x_line, x_line, linestyle="--", color="black", linewidth=1)


    
    if use_log_scale:
        plt.xscale("log")
        plt.yscale("log")

    fontsize=15
    plt.xlabel("TPV_train", fontsize=fontsize)
    plt.ylabel("TPV_test", fontsize=fontsize)
    # plt.title("TPV Trace Stability Scatter")

    cb = plt.colorbar()
    cb.set_label("Generalization gap  (L_test - L_train)")
    
    
    plt.legend()#title="# training samples")
    plt.tight_layout()
    
    if save_name is not None:
        os.makedirs(f'{results_path}/plots', exist_ok=True)
        plt.savefig(f'{results_path}/plots/{save_name}.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run experiments (resumable)
    main()

    results_path = 'results'
    exp_name = 'tpv_synthetic_results'

    TAYLOR_REL_ERR_TOL = 0.001
    cmap='Spectral'
    plot_unified_tpv_scatter(results_path, n_train_exc=None, mode='n', save_name='tpv_synth_universal_scatter_vary-n_w-256', exp_name=exp_name,
                            TAYLOR_REL_ERR_TOL=TAYLOR_REL_ERR_TOL, cmap=cmap)
    plot_unified_tpv_scatter(results_path, n_train_exc=None, mode='w', save_name='tpv_synth_universal_scatter_vary-w_n-1k', exp_name=exp_name,
                            TAYLOR_REL_ERR_TOL=TAYLOR_REL_ERR_TOL, cmap=cmap)
