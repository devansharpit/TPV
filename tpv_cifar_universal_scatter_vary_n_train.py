"""
TPV CIFAR experiment varying the number of training samples (n_train),
with:
  - Label-noise TPV (Gaussian noise on logits + MSE)
  - SGD-noise TPV (clean labels, stochastic mini-batch sampling)

Architectures: CIFAR ResNet family from chenyaofo/pytorch-cifar-models
(acts as a fixed architecture; we vary n_train only here).

Example:
  python tpv_cifar_universal_scatter_vary_n_train.py \
      --dataset c10 \
      --savefile tpv_cifar10_vary_n_train_results
"""

import os
import copy
import pickle as pkl
from typing import Tuple, List, Dict, Any, Optional

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(
        description="TPV CIFAR experiment varying number of training samples."
    )
    parser.add_argument(
        "--savefile",
        type=str,
        default="tpv_cifar10_vary_n_train_results",
        help="Base path (without extension) to save results and plots.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c10",
        choices=["c10", "c100"],
        help="Which dataset: c10 (CIFAR-10) or c100 (CIFAR-100)",
    )
    return parser.parse_args()

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
    return penalty # num_params

# -------------------------------------------------------------
# Dataset wrapper for noisy logit targets (label-noise TPV)
# -------------------------------------------------------------


class LogitNoiseDataset(Dataset):
    """
    Wraps a base dataset but replaces labels with noisy logit targets.
    Used only for label-noise TPV (Gaussian noise in logit space).
    """

    def __init__(self, base_dataset: Dataset, noisy_logits: torch.Tensor):
        assert len(base_dataset) == noisy_logits.shape[0]
        self.base_dataset = base_dataset
        self.noisy_logits = noisy_logits

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]  # ignore original label
        y = self.noisy_logits[idx]
        return x, y


# -------------------------------------------------------------
# Basic evaluation / TPV helpers
# -------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on (cross-entropy loss, accuracy) over the given dataset.
    Always uses clean integer labels provided by the dataset.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(dim=1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def compute_reference_outputs(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    n_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute f*(x) logits and probs for each x in dataset (up to n_max),
    using the reference (clean) model.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_logits = []
    all_probs = []
    total = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)

            b = x.size(0)
            if n_max is not None and total + b > n_max:
                b_keep = n_max - total
                logits = logits[:b_keep]
                probs = probs[:b_keep]
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                total += b_keep
                break
            else:
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                total += b

    if total == 0:
        raise RuntimeError("compute_reference_outputs: no samples were processed.")

    f_star_logits = torch.cat(all_logits, dim=0)
    f_star_probs = torch.cat(all_probs, dim=0)
    return f_star_logits, f_star_probs


def compute_empirical_tpv_logits_and_probs(
    model: nn.Module,
    dataset: Dataset,
    f_star_logits: torch.Tensor,
    f_star_probs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    n_max: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Empirical TPV in logits-space and prob-space:
        TPV_logits = (1/n) sum_i ||f(x_i) - f*(x_i)||_2^2
        TPV_probs  = (1/n) sum_i ||p(x_i) - p*(x_i)||_2^2
    where p = softmax(f).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sum_sq_diff_logits = 0.0
    sum_sq_diff_probs = 0.0
    total = 0

    idx_start = 0  # index into f_star tensors

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)

            b = x.size(0)
            idx_end = idx_start + b
            f_logits = f_star_logits[idx_start:idx_end].to(device)
            f_probs = f_star_probs[idx_start:idx_end].to(device)

            diff_logits = logits - f_logits
            diff_probs = probs - f_probs

            sq_diff_logits = diff_logits.pow(2).sum(dim=1)  # (b,)
            sq_diff_probs = diff_probs.pow(2).sum(dim=1)    # (b,)

            sum_sq_diff_logits += sq_diff_logits.sum().item()
            sum_sq_diff_probs += sq_diff_probs.sum().item()

            total += b
            idx_start = idx_end

            if n_max is not None and total >= n_max:
                break

    if total == 0:
        return 0.0, 0.0

    tpv_logits = sum_sq_diff_logits / total
    tpv_probs = sum_sq_diff_probs / total
    return tpv_logits, tpv_probs


# -------------------------------------------------------------
# Label-noise training: Gaussian noise on logits + MSE
# -------------------------------------------------------------


def train_model_with_logit_noise(
    base_model,
    train_dataset_tpv,
    f_star_train_logits,
    noise_std,
    n_epochs,
    lr,
    weight_decay,
    momentum,
    batch_size,
    device,
    prox_lambda=0.0,
):
    model = copy.deepcopy(base_model).to(device)
    model.eval()  # keep BN/Dropout behavior fixed

    # Reference params for optional proximity penalty
    ref_state_dict = base_model.state_dict()

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    # --- NEW: sample logit noise ONCE for the whole dataset (this run) ---
    with torch.no_grad():
        noise_all = torch.randn_like(f_star_train_logits) * noise_std
        noisy_logits_all = f_star_train_logits + noise_all

    # Dataset with fixed noisy logits
    noisy_dataset = LogitNoiseDataset(train_dataset_tpv, noisy_logits_all)

    # Deterministic loader: no shuffle, no augmentation
    loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(n_epochs):
        model.eval()
        mse_loss_epoch = 0
        cnt = 0
        for x, noisy_logits_batch in loader:
            x = x.to(device)
            noisy_logits_batch = noisy_logits_batch.to(device)

            optimizer.zero_grad()
            logits_pred = model(x)
            mse = criterion(logits_pred, noisy_logits_batch)
            mse_loss_epoch += mse.item() * x.size(0)
            cnt += x.size(0)

            if prox_lambda > 0.0:
                prox_penalty = compute_proximity_penalty(model, ref_state_dict)
                loss = mse + prox_lambda * prox_penalty
            else:
                loss = mse

            loss.backward()
            optimizer.step()
        mse_loss_epoch /= cnt
        if epoch  == 0 or epoch == n_epochs - 1:
            print(
                f"    [Label-noise epoch {epoch+1}/{n_epochs}] "
                f"MSE Loss={mse_loss_epoch:.6f}"
            )

    return model


# -------------------------------------------------------------
# SGD-noise training: clean labels, stochastic mini-batches
# -------------------------------------------------------------


def train_model_with_sgd_noise(
    base_model: nn.Module,
    train_dataset: Dataset,
    train_dataset_tpv: Dataset,
    test_dataset_tpv: Dataset,
    test_dataset_full: Dataset,
    f_star_train_logits: torch.Tensor,
    f_star_train_probs: torch.Tensor,
    f_star_test_logits: torch.Tensor,
    f_star_test_probs: torch.Tensor,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    batch_size: int,
    batch_size_eval: int,
    device: torch.device,
    n_train_tpv: Optional[int] = None,
    n_test_tpv: Optional[int] = None,
    prox_lambda: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """
    Train a copy of base_model on CLEAN labels using SGD (mini-batch sampling)
    to realize SGD noise, with BN/Dropout frozen (eval-mode).

    We take snapshots at the end of each epoch, compute TPV on train/test TPV
    subsets, and average those TPV values over epochs. Finally we return:
      - mean TPV_train_logits, mean TPV_test_logits
      - mean TPV_train_probs,  mean TPV_test_probs
      - final train CE loss (clean labels), final test CE loss (clean labels)
    """
    model = copy.deepcopy(base_model).to(device)
    model.eval()  # freeze BN/Dropout behavior

    # Save reference parameters for proximity penalty
    ref_state_dict = base_model.state_dict()

    # SGD noise arises from mini-batch sampling (shuffle=True) on clean data
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    tpv_train_logits_sum = 0.0
    tpv_test_logits_sum = 0.0
    tpv_train_probs_sum = 0.0
    tpv_test_probs_sum = 0.0
    snapshots = 0

    def _subset_dataset(full_dataset: Dataset, n_max: Optional[int]):
        if n_max is None:
            return full_dataset
        n_total = len(full_dataset)
        n_use = min(n_max, n_total)
        indices = np.arange(n_total)[:n_use]
        return Subset(full_dataset, indices)

    for epoch in range(n_epochs):
        # Even though model is in eval() mode, gradients still flow and params update
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            mse = criterion(logits, y)
            if prox_lambda > 0.0:
                prox_penalty = compute_proximity_penalty(model, ref_state_dict)
                loss = mse + prox_lambda * prox_penalty
            else:
                loss = mse

            loss.backward()
            optimizer.step()

        # Take snapshot at end of each epoch
        model.eval()

        train_subset_for_tpv = _subset_dataset(train_dataset_tpv, n_train_tpv)
        test_subset_for_tpv = _subset_dataset(test_dataset_tpv, n_test_tpv)

        print(
            f"    [SGD snapshot epoch {epoch+1}/{n_epochs}] "
            f"Computing empirical TPV on train subset..."
        )
        tpv_train_logits_epoch, tpv_train_probs_epoch = compute_empirical_tpv_logits_and_probs(
            model,
            train_subset_for_tpv,
            f_star_train_logits,
            f_star_train_probs,
            batch_size_eval,
            device,
            n_max=n_train_tpv,
        )
        print(
            f"    [SGD snapshot epoch {epoch+1}/{n_epochs}] "
            f"TPV_train_logits={tpv_train_logits_epoch:.6e}, "
            f"TPV_train_probs={tpv_train_probs_epoch:.6e}"
        )

        print(
            f"    [SGD snapshot epoch {epoch+1}/{n_epochs}] "
            f"Computing empirical TPV on test subset..."
        )
        tpv_test_logits_epoch, tpv_test_probs_epoch = compute_empirical_tpv_logits_and_probs(
            model,
            test_subset_for_tpv,
            f_star_test_logits,
            f_star_test_probs,
            batch_size_eval,
            device,
            n_max=n_test_tpv,
        )
        print(
            f"    [SGD snapshot epoch {epoch+1}/{n_epochs}] "
            f"TPV_test_logits={tpv_test_logits_epoch:.6e}, "
            f"TPV_test_probs={tpv_test_probs_epoch:.6e}"
        )

        tpv_train_logits_sum += tpv_train_logits_epoch
        tpv_test_logits_sum += tpv_test_logits_epoch
        tpv_train_probs_sum += tpv_train_probs_epoch
        tpv_test_probs_sum += tpv_test_probs_epoch
        snapshots += 1

    if snapshots == 0:
        raise RuntimeError("train_model_with_sgd_noise: no snapshots collected.")

    tpv_train_logits_mean = tpv_train_logits_sum / snapshots
    tpv_test_logits_mean = tpv_test_logits_sum / snapshots
    tpv_train_probs_mean = tpv_train_probs_sum / snapshots
    tpv_test_probs_mean = tpv_test_probs_sum / snapshots

    # Generalization gap of the SGD-trained model on CLEAN labels
    train_loss_clean, _ = evaluate_model(
        model, train_dataset, batch_size_eval, device
    )
    test_loss_clean, _ = evaluate_model(
        model, test_dataset_full, batch_size_eval, device
    )

    return (
        tpv_train_logits_mean,
        tpv_test_logits_mean,
        tpv_train_probs_mean,
        tpv_test_probs_mean,
        train_loss_clean,
        test_loss_clean,
    )


# -------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------


def main(args):
    

    # Reproducibility
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    # Vary number of training samples (like synthetic experiment)
    N_TRAIN_CHOICES = [1, 10, 10000]

    # Label-noise TPV config (Gaussian logit noise)
    LOGIT_NOISE_STD_LIST = [0.1]  
    R_LABEL = 5                     # runs per (arch, noise_std, n_train)
    N_EPOCHS_NOISY_LABEL = 10
    LR_NOISY_LABEL = 1e-4
    WEIGHT_DECAY_LABEL = 0.0
    MOMENTUM_LABEL = 0.9
    BATCH_SIZE_LABEL = 256
    PROX_LAMBDA_LABEL = 0# 1e-2       # mean-normalized proximity penalty

    # SGD-noise config (clean labels, stochastic mini-batches)
    SGD_LR_LIST = [1e-4, 5e-5] 
    SGD_BATCH_SIZE_LIST = [128, 256]
    N_EPOCHS_SGD_NOISY = 10
    WEIGHT_DECAY_SGD = 0.0
    MOMENTUM_SGD = 0.9

    # TPV subset sizes
    N_TRAIN_TPV = 10000
    N_TEST_TPV = 10000

    DATA_ROOT = "./data"
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_PATH = f"{RESULTS_DIR}/{args.savefile}.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Deterministic transform (no augmentation) for TPV experiments
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # Load CIFAR datasets
    if args.dataset == "c10":
        NUM_CLASSES = 10
        ARCH_SPECS = [
            ("resnet20", "cifar10_resnet20"),
            ("resnet32", "cifar10_resnet32"),
            ("resnet44", "cifar10_resnet44"),
            ("resnet56", "cifar10_resnet56"),
        ]
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=transform_eval
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform_eval
        )
    elif args.dataset == "c100":
        NUM_CLASSES = 100
        ARCH_SPECS = [
            ("resnet20", "cifar100_resnet20"),
            ("resnet32", "cifar100_resnet32"),
            ("resnet44", "cifar100_resnet44"),
            ("resnet56", "cifar100_resnet56"),
        ]
        train_dataset = torchvision.datasets.CIFAR100(
            root=DATA_ROOT, train=True, download=True, transform=transform_eval
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=DATA_ROOT, train=False, download=True, transform=transform_eval
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_train_total = len(train_dataset)
    n_test_total = len(test_dataset)
    print(f"Train set size: {n_train_total}, Test set size: {n_test_total}")

    # Clean labels tensor for full train set
    if hasattr(train_dataset, "targets"):
        y_clean_train = torch.tensor(train_dataset.targets, dtype=torch.long)
    else:
        y_clean_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

    # Test TPV subset
    test_indices_tpv = np.arange(n_test_total)
    if N_TEST_TPV is not None and N_TEST_TPV < n_test_total:
        test_indices_tpv = test_indices_tpv[:N_TEST_TPV]
    test_dataset_tpv = Subset(test_dataset, indices=test_indices_tpv)

    # ---------------------------------------------------------
    # Main loop over architectures
    # ---------------------------------------------------------

    results: Dict[Any, Any] = {}

    for (short_name, hub_name) in ARCH_SPECS:
        print("\n==============================================")
        print(f"Architecture: {short_name} ({hub_name})")
        print("Loading pretrained clean model from torch.hub...")

        base_model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            hub_name,
            pretrained=True,
        ).to(device)
        base_model.eval()

        # Evaluate clean model on full clean TRAIN and TEST to get generalization gap
        print("Evaluating clean reference model on full train/test...")
        clean_train_loss, clean_train_acc = evaluate_model(
            base_model, train_dataset, BATCH_SIZE_LABEL, device
        )
        clean_test_loss, clean_test_acc = evaluate_model(
            base_model, test_dataset, BATCH_SIZE_LABEL, device
        )
        gen_gap_clean = clean_test_loss - clean_train_loss

        print(
            f"  Clean TRAIN  loss={clean_train_loss:.4f}, acc={clean_train_acc*100:.2f}%"
        )
        print(
            f"  Clean TEST   loss={clean_test_loss:.4f}, acc={clean_test_acc*100:.2f}%"
        )
        print(
            f"  Clean generalization gap (CE) = {gen_gap_clean:.4f}"
        )

        # Store clean reference info
        results[("clean_reference", hub_name)] = {
            "arch_name": short_name,
            "hub_name": hub_name,
            "L_train_clean": clean_train_loss,
            "L_test_clean": clean_test_loss,
            "clean_train_acc": clean_train_acc,
            "clean_test_acc": clean_test_acc,
            "gen_gap_clean": gen_gap_clean,
            "train_tpv_n": N_TRAIN_TPV,
            "test_tpv_n": N_TEST_TPV,
        }

        for n_train_choice in N_TRAIN_CHOICES:
            print("\n----------------------------------------------")
            print(f"  Using n_train = {n_train_choice} training examples")

            # Randomly select training indices ONCE per (arch, n_train)
            indices_train_sub = np.random.choice(
                n_train_total, size=n_train_choice, replace=False
            )
            y_clean_sub = y_clean_train[indices_train_sub]
            base_train_dataset_sub = Subset(train_dataset, indices=indices_train_sub)

            # Train TPV subset: for label-noise TPV we want TPV subset == training subset
            n_train_tpv = min(n_train_choice, N_TRAIN_TPV)
            indices_train_tpv_sub = np.arange(n_train_choice)[:n_train_tpv]
            train_dataset_tpv = Subset(base_train_dataset_sub, indices=indices_train_tpv_sub)

            assert (
                n_train_tpv == n_train_choice
            ), "For label-noise TPV we currently assume n_train_tpv == n_train_choice."

            print("  Computing reference outputs f*(x) on train TPV subset...")
            f_star_train_logits, f_star_train_probs = compute_reference_outputs(
                base_model, train_dataset_tpv, BATCH_SIZE_LABEL, device, n_max=n_train_tpv
            )
            print("  Computing reference outputs f*(x) on test TPV subset...")
            f_star_test_logits, f_star_test_probs = compute_reference_outputs(
                base_model, test_dataset_tpv, BATCH_SIZE_LABEL, device, n_max=N_TEST_TPV
            )

            # ----------------------------
            # Label-noise TPV experiments (Gaussian logit noise)
            # ----------------------------
            for noise_std in LOGIT_NOISE_STD_LIST:
                print(
                    f"\n  [Label noise] noise_std={noise_std}, "
                    f"n_train={n_train_choice} for architecture {short_name}"
                )

                for run_idx in range(R_LABEL):
                    print(f"    Run {run_idx+1}/{R_LABEL}")

                    # Construct noisy-logit dataset for this run on the same train TPV subset
                    # We draw Gaussian noise inside train_model_with_logit_noise; here
                    # we just pass the base dataset (noisy logits are internal).
                    model_noisy = train_model_with_logit_noise(
                        base_model,
                        train_dataset_tpv,
                        f_star_train_logits,
                        noise_std=noise_std,
                        n_epochs=N_EPOCHS_NOISY_LABEL,
                        lr=LR_NOISY_LABEL,
                        weight_decay=WEIGHT_DECAY_LABEL,
                        momentum=MOMENTUM_LABEL,
                        batch_size=BATCH_SIZE_LABEL,
                        device=device,
                        prox_lambda=PROX_LAMBDA_LABEL,
                    )

                    print(
                        "      Computing empirical TPV (logits & probs) on train subset..."
                    )
                    tpv_train_logits_r, tpv_train_probs_r = compute_empirical_tpv_logits_and_probs(
                        model_noisy,
                        train_dataset_tpv,
                        f_star_train_logits,
                        f_star_train_probs,
                        BATCH_SIZE_LABEL,
                        device,
                        n_max=n_train_tpv,
                    )
                    print(
                        "      Computing empirical TPV (logits & probs) on test subset..."
                    )
                    tpv_test_logits_r, tpv_test_probs_r = compute_empirical_tpv_logits_and_probs(
                        model_noisy,
                        test_dataset_tpv,
                        f_star_test_logits,
                        f_star_test_probs,
                        BATCH_SIZE_LABEL,
                        device,
                        n_max=N_TEST_TPV,
                    )

                    # Generalization of the noisy-trained model on CLEAN labels
                    train_loss_clean, _ = evaluate_model(
                        model_noisy, base_train_dataset_sub, BATCH_SIZE_LABEL, device
                    )
                    test_loss_clean, _ = evaluate_model(
                        model_noisy, test_dataset, BATCH_SIZE_LABEL, device
                    )
                    gen_gap_noisy = test_loss_clean - train_loss_clean

                    print(
                        f"      [Label-noise run {run_idx+1}] "
                        f"TPV_train_logits={tpv_train_logits_r:.6e}, "
                        f"TPV_test_logits={tpv_test_logits_r:.6e}, "
                        f"TPV_train_probs={tpv_train_probs_r:.6e}, "
                        f"TPV_test_probs={tpv_test_probs_r:.6e}, "
                        f"L_train_clean={train_loss_clean:.4f}, "
                        f"L_test_clean={test_loss_clean:.4f}"
                    )

                    results[("label_noise", hub_name, n_train_choice, noise_std, run_idx)] = {
                        "source": "label_noise",
                        "arch_name": short_name,
                        "hub_name": hub_name,
                        "noise_std": noise_std,
                        "n_train": n_train_choice,
                        "lr": LR_NOISY_LABEL,
                        "batch_size": BATCH_SIZE_LABEL,
                        "n_epochs": N_EPOCHS_NOISY_LABEL,
                        "tpv_train_logits": tpv_train_logits_r,
                        "tpv_test_logits": tpv_test_logits_r,
                        "tpv_train_probs": tpv_train_probs_r,
                        "tpv_test_probs": tpv_test_probs_r,
                        "L_train_noisy": train_loss_clean,
                        "L_test_noisy": test_loss_clean,
                        "gen_gap_noisy": gen_gap_noisy,
                        "train_tpv_n": n_train_tpv,
                        "test_tpv_n": N_TEST_TPV,
                    }

            # ----------------------------
            # SGD-noise TPV experiments (clean labels)
            # ----------------------------
            print(
                f"\n  [SGD noise] n_train={n_train_choice} "
                f"(clean labels, varying batch_size and lr)"
            )

            for sgd_lr in SGD_LR_LIST:
                for sgd_bs in SGD_BATCH_SIZE_LIST:
                    print(
                        f"    [SGD] lr={sgd_lr}, batch_size={sgd_bs}, "
                        f"epochs={N_EPOCHS_SGD_NOISY}"
                    )

                    (
                        tpv_train_logits_sgd,
                        tpv_test_logits_sgd,
                        tpv_train_probs_sgd,
                        tpv_test_probs_sgd,
                        train_loss_sgd,
                        test_loss_sgd,
                    ) = train_model_with_sgd_noise(
                        base_model=base_model,
                        train_dataset=base_train_dataset_sub,
                        train_dataset_tpv=train_dataset_tpv,
                        test_dataset_tpv=test_dataset_tpv,
                        test_dataset_full=test_dataset,
                        f_star_train_logits=f_star_train_logits,
                        f_star_train_probs=f_star_train_probs,
                        f_star_test_logits=f_star_test_logits,
                        f_star_test_probs=f_star_test_probs,
                        n_epochs=N_EPOCHS_SGD_NOISY,
                        lr=sgd_lr,
                        weight_decay=WEIGHT_DECAY_SGD,
                        momentum=MOMENTUM_SGD,
                        batch_size=sgd_bs,
                        batch_size_eval=BATCH_SIZE_LABEL,
                        device=device,
                        n_train_tpv=n_train_tpv,
                        n_test_tpv=N_TEST_TPV,
                    )

                    gen_gap_sgd = test_loss_sgd - train_loss_sgd

                    print(
                        f"      [SGD run] "
                        f"TPV_train_logits={tpv_train_logits_sgd:.6e}, "
                        f"TPV_test_logits={tpv_test_logits_sgd:.6e}, "
                        f"TPV_train_probs={tpv_train_probs_sgd:.6e}, "
                        f"TPV_test_probs={tpv_test_probs_sgd:.6e}, "
                        f"L_train_clean={train_loss_sgd:.4f}, "
                        f"L_test_clean={test_loss_sgd:.4f}"
                    )

                    results[("sgd_noise", hub_name, n_train_choice, sgd_lr, sgd_bs)] = {
                        "source": "sgd_noise",
                        "arch_name": short_name,
                        "hub_name": hub_name,
                        "n_train": n_train_choice,
                        "sgd_lr": sgd_lr,
                        "batch_size": sgd_bs,
                        "n_epochs": N_EPOCHS_SGD_NOISY,
                        "tpv_train_logits": tpv_train_logits_sgd,
                        "tpv_test_logits": tpv_test_logits_sgd,
                        "tpv_train_probs": tpv_train_probs_sgd,
                        "tpv_test_probs": tpv_test_probs_sgd,
                        "L_train_noisy": train_loss_sgd,
                        "L_test_noisy": test_loss_sgd,
                        "gen_gap_noisy": gen_gap_sgd,
                        "train_tpv_n": n_train_tpv,
                        "test_tpv_n": N_TEST_TPV,
                    }

    # Save all results
    with open(RESULTS_PATH, "wb") as f:
        pkl.dump(results, f)
    print(f"\nSaved all TPV results to {RESULTS_PATH}")


def plot_unified_tpv_scatter(results_path: str,
                             args,
                             space: str = "probs",
                             use_log_scale: bool = True,
                             savepath: str = None, n_train_exc=[], exp_name=None):
    """
    Make a single scatter plot:

        x-axis: TPV_train (in chosen space)
        y-axis: TPV_test  (in chosen space)
        color:  generalization gap (L_test - L_train) of the noisy-trained model
        marker: number of training samples (n_train)

    space: "probs" or "logits"
    """
    assert space in ["probs", "logits"], "space must be 'probs' or 'logits'"

    with open(results_path+f'/{exp_name}.pkl', "rb") as f:
        results = pkl.load(f)

    tpv_train_all = []
    tpv_test_all = []
    gen_gap_all = []
    ntrain_all = []

    for key, val in results.items():
        if not isinstance(key, tuple) or not key:
            continue

        tag = key[0]
        if tag not in ["label_noise", "sgd_noise"]:
            continue

        if space == "probs":
            tpv_train = val["tpv_train_probs"]
            tpv_test = val["tpv_test_probs"]
        else:
            tpv_train = val["tpv_train_logits"]
            tpv_test = val["tpv_test_logits"]

        tpv_train_all.append(tpv_train)
        tpv_test_all.append(tpv_test)
        L_train = val["L_train_noisy"]
        L_test = val["L_test_noisy"]
        gen_gap_all.append(L_test - L_train)
        ntrain_all.append(val["n_train"])

    tpv_train_all = np.array(tpv_train_all)
    tpv_test_all = np.array(tpv_test_all)
    gen_gap_all = np.array(gen_gap_all)
    ntrain_all = np.array(ntrain_all)

    if tpv_train_all.size == 0:
        print("No TPV entries found to plot.")
        return

    # --- Figure & axis ---
    fig, ax = plt.subplots(figsize=(4, 3))

    # Markers for different n_train values
    markers_by_n = {
        1: 'o',
        10: "+",
        10000: "^",
    }

    # Shared colormap range for all points
    vmin = np.min(gen_gap_all)
    vmax = np.max(gen_gap_all)
    cmap = "viridis"

    sc = None
    min_val_, max_val_ = np.inf, 0
    for n_train in sorted(np.unique(ntrain_all)):
        if n_train in n_train_exc: continue
        mask = (ntrain_all == n_train)
        if not np.any(mask):
            continue

        sc = ax.scatter(
            tpv_train_all[mask],
            tpv_test_all[mask],
            c=gen_gap_all[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=markers_by_n.get(int(n_train), "o"),
            s=40,
            alpha=0.8,
            edgecolors="none",
            label=f"n_train={int(n_train)}",
        )
        min_val_ = min(min_val_, min(tpv_train_all[mask].min(), tpv_test_all[mask].min()))
        max_val_ = max(max_val_, max(tpv_train_all[mask].max(), tpv_test_all[mask].max()))

    # Diagonal y = x
    min_val = min(tpv_train_all.min(), tpv_test_all.min())
    max_val = max(tpv_train_all.max(), tpv_test_all.max())

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

    
    ax.plot(
        [min_val_, max_val_],
        [min_val_, max_val_],
        linestyle="--",
        linewidth=1.0,
        color="black",
    )

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    label_space = "probabilities" if space == "probs" else "logits"
    ax.set_xlabel(f"TPV$_{{\\text{{train}}}}$", fontsize=12)
    ax.set_ylabel(f"TPV$_{{\\text{{test}}}}$", fontsize=12)
    ax.set_title(f"vary n_train", fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=10)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Generalization gap  $L_{\\text{test}} - L_{\\text{train}}$", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    ax.legend(
        # title="# training samples",
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        frameon=True,
    )

    fig.tight_layout()
    os.makedirs(f'{results_path}/plots', exist_ok=True)
    plt.savefig(f'{results_path}/plots/tpv_{args.dataset}_universal_scatter_vary_n_train.pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    args = get_args()
    main(args)

    results_path = 'results'
    exp_name = args.savefile 

    plot_unified_tpv_scatter(results_path, args, space='logits', n_train_exc=[], exp_name=exp_name)
