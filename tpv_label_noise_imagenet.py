import os
import math
import numpy as np
import pickle as pkl
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader, Subset, Dataset

from imagenet_dataloader import create_imagenet_dataloaders

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


# ----------------------------
# Training (MSE on logits) using DataLoader
# ----------------------------
def train_mse_with_loader(
    model,
    train_loader,
    max_epochs=1000,
    train_mse_thres=0.001,
    lr=1e-3,
    wd=0.0,
    momentum=0.9,
    opt="sgd",
    print_stats=False,
    ref_state_dict=None,
    proximity_lambda=0.0,
    accumulate_gradients=False,
    warmup_first_epoch=True,
):
    """
    Train model to regress its logits to *noisy* logit targets using MSE loss.
    The train_loader must yield:
        (images, noisy_logits, teacher_logits, labels, indices)

    Training stops when the average MSE w.r.t. noisy targets on the training
    set goes below train_mse_thres or max_epochs is reached.

    Args:
        accumulate_gradients: If True, accumulate gradients over the entire epoch
                             and update parameters only once per epoch (full-batch GD).
                             If False, update parameters after each mini-batch (default).

    Returns: (final_loss, epochs_trained)
    """
    # model.train()
    model.eval()
    if opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    criterion = nn.MSELoss(reduction="sum") # 

    epochs_trained = 0
    num_batches = len(train_loader)

    total_loss_list = []

    total_loss = 0.0
    total_samples = 0
    for batch_idx, (images, noisy_logits, teacher_logits, labels, indices) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        noisy_logits = noisy_logits.to(device, non_blocking=True)
        logits = model(images)
        mse_loss = criterion(logits, noisy_logits)
        total_loss += mse_loss.item()
        total_samples += images.size(0)
    avg_mse = total_loss / (max(total_samples, 1))
    total_loss_list.append(avg_mse)
    print(f"[MSE] Initial train MSE (noisy targets)={avg_mse}")

    pbar = tqdm(range(max_epochs), desc="Training")
    for epoch in pbar:
        epochs_trained = epoch + 1
        # model.train()
        model.eval()
        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        # Zero gradients at the start of epoch if accumulating
        if accumulate_gradients:
            optimizer.zero_grad()

        for batch_idx, (images, noisy_logits, teacher_logits, labels, indices) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            noisy_logits = noisy_logits.to(device, non_blocking=True)

            # Zero gradients for mini-batch mode
            if not accumulate_gradients:
                optimizer.zero_grad()

            logits = model(images)
            mse_loss = criterion(logits, noisy_logits)

            total_loss_ = mse_loss

            # Add proximity penalty if reference parameters provided
            if ref_state_dict is not None and proximity_lambda > 0:
                prox_penalty = compute_proximity_penalty(model, ref_state_dict)
                total_loss_ = mse_loss + proximity_lambda * prox_penalty

            if not torch.isfinite(total_loss_):
                print(f"[train_mse_with_loader] Non-finite loss at epoch {epoch}: {total_loss_.item()}")
                return None, epochs_trained, total_loss_list

            batch_size = images.size(0)
            
            if accumulate_gradients:
                # Normalize gradient by total number of samples for full-batch GD
                # We'll divide by total_batches at the end of epoch
                total_loss_.backward()
            else:
                # Standard mini-batch: backward and step immediately
                total_loss_.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            total_loss += mse_loss.item()
            total_samples += batch_size
            total_batches+=1

        # For gradient accumulation mode: normalize gradients and update once per epoch
        if accumulate_gradients:
            # Normalize accumulated gradients by the number of samples
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(total_batches)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Average MSE per sample w.r.t noisy targets
        avg_mse = total_loss / max(total_samples, 1)
        total_loss_list.append(avg_mse)

        # Update progress bar with current MSE
        pbar.set_postfix({"MSE": f"{avg_mse}"})

        if (print_stats and (epoch % 5 == 0 or epoch == max_epochs - 1)) or epoch == 0 or epoch == max_epochs - 1:
            prox_penalty = compute_proximity_penalty(model, ref_state_dict)
            print(f"[MSE] epoch {epoch:03d} | train MSE (noisy targets)={avg_mse} | proximity penalty={prox_penalty.item():.8f}")

        if len(total_loss_list)>1 and total_loss_list[-1]>total_loss_list[0]:
            print(f"[WARNING] Loss exploded at epoch {epoch}: {avg_mse}")
            # break
        if avg_mse < train_mse_thres:
            pbar.close()
            break
    
    if epochs_trained == max_epochs:
        pbar.close()
    
    print(f"[MSE] Final train MSE (noisy targets)={avg_mse}")
    return None, epochs_trained, total_loss_list

# ----------------------------
# Eval helper: compute TPV and losses batchwise
# ----------------------------
def compute_run_stats(
    model,
    train_eval_loader,
    val_eval_loader,
    num_classes,
    noise_std,
    ce_criterion=None,
):
    """
    Given a trained model for a single run, compute:
        - sum_sqdiff_train: sum_i || f_hat(x_i) - f_star(x_i) ||^2 over train
        - sum_sqdiff_val:   sum_i || f_hat(x_i) - f_star(x_i) ||^2 over val
        - run_mse_train_noisy_targets
        - run_mse_train_clean_logits
        - run_mse_val_clean_logits
        - run_ce_val_clean_labels

    All computed batchwise to be memory efficient.

    train_eval_loader must yield:
        (images, noisy_logits, teacher_logits, labels, indices)
    val_eval_loader must yield:
        (images, teacher_logits, labels, indices)
    """
    if ce_criterion is None:
        ce_criterion = nn.CrossEntropyLoss() # reduction="sum"

    mse_criterion = nn.MSELoss() # reduction="sum"

    model.eval()

    # Train stats
    sum_sqdiff_train = 0.0
    sum_mse_train_noisy = 0.0
    sum_mse_train_clean = 0.0
    n_train_batches = 0

    with torch.no_grad():
        for images, noisy_logits, teacher_logits, labels, indices in train_eval_loader:
            images = images.to(device, non_blocking=True)
            noisy_logits = noisy_logits.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)

            logits = model(images)
            diff = logits - teacher_logits
            sq = torch.sum(diff ** 2, dim=1)  # (batch,)
            sum_sqdiff_train += sq.sum().item()

            sum_mse_train_noisy += mse_criterion(logits, noisy_logits).item()
            sum_mse_train_clean += mse_criterion(logits, teacher_logits).item()
            n_train_batches += 1

    # Val stats
    sum_sqdiff_val = 0.0
    sum_mse_val_clean = 0.0
    sum_ce_val = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for images, teacher_logits, labels, indices in val_eval_loader:
            images = images.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            diff = logits - teacher_logits
            sq = torch.sum(diff ** 2, dim=1)
            sum_sqdiff_val += sq.sum().item()

            sum_mse_val_clean += mse_criterion(logits, teacher_logits).item()
            sum_ce_val += ce_criterion(logits, labels).item()
            n_val_batches += 1

    run_stats = {
        "sum_sqdiff_train": sum_sqdiff_train,
        "sum_sqdiff_val": sum_sqdiff_val,
        "run_mse_train_noisy_targets": sum_mse_train_noisy / max(n_train_batches, 1),
        "run_mse_train_clean_logits": sum_mse_train_clean / max(n_train_batches, 1),
        "run_mse_val_clean_logits": sum_mse_val_clean / max(n_val_batches, 1),
        "run_ce_val_clean_labels": sum_ce_val / max(n_val_batches, 1),
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
    }
    return run_stats


# ----------------------------
# Datasets that attach logits/labels to subsets
# ----------------------------
class TrainSubsetWithLogits(Dataset):
    """
    Wrap a Subset (of ImageNet train) and attach:
        - teacher_logits_train: (N, K) CPU tensor
        - noisy_logits_train:   (N, K) CPU tensor for a given run
        - labels_train:         (N,)   CPU tensor
    We keep all tensors on CPU and only move per-batch to GPU.
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
        teacher_logits = self.teacher_logits[idx]
        noisy_logits = self.noisy_logits[idx]
        label = self.labels[idx]
        return img, noisy_logits, teacher_logits, label, idx


class TrainSubsetForEval(Dataset):
    """
    For evaluation of a single run on training set:
        yields (img, noisy_logits, teacher_logits, label, idx)
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
        return img, self.noisy_logits[idx], self.teacher_logits[idx], self.labels[idx], idx


class ValSubsetWithLogits(Dataset):
    """
    Wrap a Subset (of ImageNet val) and attach:
        - teacher_logits_val: (N, K) CPU tensor
        - labels_val:         (N,)   CPU tensor
    """
    def __init__(self, subset, teacher_logits, labels):
        self.subset = subset
        self.teacher_logits = teacher_logits
        self.labels = labels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        teacher_logits = self.teacher_logits[idx]
        label = self.labels[idx]
        return img, teacher_logits, label, idx


# ----------------------------
# Compute teacher logits + labels for a subset
# ----------------------------
def compute_teacher_logits_and_labels(model, subset, batch_size, num_classes):
    """
    Compute teacher logits and labels for a subset using model, batchwise.

    Returns:
        teacher_logits: (N, num_classes) float32 CPU tensor
        labels:         (N,)            long   CPU tensor
        baseline_mse:   MSE of model logits vs itself (should be ~0)
        baseline_acc:   top-1 accuracy (only for val subset; for train subset
                        you can ignore or treat as train accuracy on subset)
    """
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # We'll infer N from subset
    N = len(subset)
    teacher_logits = torch.empty((N, num_classes), dtype=torch.float32)
    labels = torch.empty((N,), dtype=torch.long)

    mse_criterion = nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()
    # model.train()
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            bsz = images.size(0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)  # (bsz, K)
            preds = torch.argmax(logits, dim=1)

            teacher_logits[offset:offset+bsz].copy_(logits.cpu())
            labels[offset:offset+bsz].copy_(targets.cpu())
            offset += bsz

            total_mse += mse_criterion(logits, logits).item()  # should be 0
            total_correct += (preds == targets).sum().item()
            total_samples += bsz

    baseline_mse = total_mse / max(total_samples, 1)
    baseline_acc = total_correct / max(total_samples, 1)

    return teacher_logits, labels, baseline_mse, baseline_acc

def estimate_empirical_TPV_logit_noise(
    model_ctor,
    model_name,
    noise_std,
    R,
    train_subset,
    val_subset,
    teacher_train_logits,
    teacher_val_logits,
    labels_train,
    labels_val,
    base_state_dict,
    max_epochs_noisy=1000,
    train_mse_thres=0.001,
    lr_noisy=1e-3,
    momentum=0.9,
    wd=0.0,
    opt="sgd",
    proximity_lambda=0.0,
    train_batch_size=32,
    eval_batch_size=64,
    accumulate_gradients=False,
):
    """
    Memory-efficient version of empirical TPV estimation:

    - For each run r:
        * sample Gaussian logit noise on teacher logits (train)
        * retrain from w* using MSE on noisy logits
        * measure deviations from teacher logits on train/val
    - TPV is average (over runs, data) of ||z_hat - z_star||^2.

    We do everything batchwise, never storing all predictions for all runs.

    IMPORTANT: We make SGD / DataLoader behavior deterministic across runs
    and let ONLY label noise differ between runs. This is done via:
        - global seeding at the top of the file (torch / np / random)
        - a dedicated DataLoader generator with a fixed seed each run
        - label noise sampling using the global torch RNG (no reseeding)
    """
    mse_train_noisy_targets_list = []
    mse_train_clean_logits_list = []
    mse_val_clean_logits_list = []
    ce_val_clean_labels_list = []

    ce_criterion = nn.CrossEntropyLoss() # reduction="sum"

    num_classes = teacher_train_logits.shape[1]
    n_train = teacher_train_logits.shape[0]
    n_val = teacher_val_logits.shape[0]

    total_sqdiff_train = 0.0
    total_sqdiff_val = 0.0

    # Dedicated generator for DataLoader shuffling; we will reset this
    # to the SAME seed each run so minibatch order is identical.
    LOADER_SEED = 12345
    total_loss_lol = []
    for r in range(R):
        print(f"      Run {r+1}/{R} ...")

        # Start from clean reference model weights
        model = model_ctor(pretrained=False).to(device)
        model.load_state_dict(base_state_dict)

        # --- LABEL NOISE (DIFFERENT ACROSS RUNS) ---
        # Use global torch RNG (seeded once at top). We DO NOT reseed here.
        eps_train = torch.randn_like(teacher_train_logits) * noise_std
        noisy_train_logits = teacher_train_logits + eps_train  # CPU tensor (N, K)

        # Training dataset for this run
        train_run_dataset = TrainSubsetWithLogits(
            subset=train_subset,
            teacher_logits=teacher_train_logits,
            noisy_logits=noisy_train_logits,
            labels=labels_train,
        )

        # --- DATALOADER (IDENTICAL ACROSS RUNS) ---
        # Use a dedicated generator, seeded to the SAME value each run,
        # so shuffle order is deterministic and identical across runs.
        loader_gen = torch.Generator()
        loader_gen.manual_seed(LOADER_SEED)

        train_run_loader = DataLoader(
            train_run_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=loader_gen,  # <- deterministic minibatch order
            num_workers=4,
            pin_memory=True,
        )

        # Train with MSE on noisy logits
        _, epochs_trained, total_loss_list = train_mse_with_loader(
            model,
            train_run_loader,
            max_epochs=max_epochs_noisy,
            train_mse_thres=train_mse_thres,
            lr=lr_noisy,
            wd=wd,
            momentum=momentum,
            opt=opt,
            print_stats=False,
            ref_state_dict=base_state_dict,
            proximity_lambda=proximity_lambda,
            accumulate_gradients=accumulate_gradients,
        )
        total_loss_lol.append(total_loss_list)
        print(f"        epochs_trained = {epochs_trained}")

        # Evaluation loaders for this run (no shuffle)
        train_eval_dataset = TrainSubsetForEval(
            subset=train_subset,
            teacher_logits=teacher_train_logits,
            noisy_logits=noisy_train_logits,
            labels=labels_train,
        )
        train_eval_loader = DataLoader(
            train_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        val_eval_dataset = ValSubsetWithLogits(
            subset=val_subset,
            teacher_logits=teacher_val_logits,
            labels=labels_val,
        )
        val_eval_loader = DataLoader(
            val_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Compute run-level stats + squared differences
        run_stats = compute_run_stats(
            model,
            train_eval_loader,
            val_eval_loader,
            num_classes=num_classes,
            noise_std=noise_std,
            ce_criterion=ce_criterion,
        )

        total_sqdiff_train += run_stats["sum_sqdiff_train"]
        total_sqdiff_val += run_stats["sum_sqdiff_val"]

        mse_train_noisy_targets_list.append(run_stats["run_mse_train_noisy_targets"])
        mse_train_clean_logits_list.append(run_stats["run_mse_train_clean_logits"])
        mse_val_clean_logits_list.append(run_stats["run_mse_val_clean_logits"])
        ce_val_clean_labels_list.append(run_stats["run_ce_val_clean_labels"])

        # Free model for this run
        del model
        torch.cuda.empty_cache()

    # Empirical TPV: mean squared deviation from clean logits over runs and samples
    empirical_TPV_train = total_sqdiff_train / (R * max(n_train, 1))
    empirical_TPV_val = total_sqdiff_val / (R * max(n_val, 1))

    mse_train_noisy_targets_mean = float(np.mean(mse_train_noisy_targets_list))
    mse_train_noisy_targets_std = float(np.std(mse_train_noisy_targets_list))

    mse_train_clean_logits_mean = float(np.mean(mse_train_clean_logits_list))
    mse_train_clean_logits_std = float(np.std(mse_train_clean_logits_list))

    mse_val_clean_logits_mean = float(np.mean(mse_val_clean_logits_list))
    mse_val_clean_logits_std = float(np.std(mse_val_clean_logits_list))

    ce_val_clean_labels_mean = float(np.mean(ce_val_clean_labels_list))
    ce_val_clean_labels_std = float(np.std(ce_val_clean_labels_list))

    return dict(
        empirical_TPV_train=empirical_TPV_train,
        empirical_TPV_val=empirical_TPV_val,
        mse_train_noisy_targets_mean=mse_train_noisy_targets_mean,
        mse_train_noisy_targets_std=mse_train_noisy_targets_std,
        mse_train_clean_logits_mean=mse_train_clean_logits_mean,
        mse_train_clean_logits_std=mse_train_clean_logits_std,
        mse_val_clean_logits_mean=mse_val_clean_logits_mean,
        mse_val_clean_logits_std=mse_val_clean_logits_std,
        ce_val_clean_labels_mean=ce_val_clean_labels_mean,
        ce_val_clean_labels_std=ce_val_clean_labels_std,
        total_loss_lol=total_loss_lol,
    )




# ----------------------------
# Main: ImageNet TPV experiment (memory efficient)
# ----------------------------
if __name__ == "__main__":
    # ------------------------
    # ImageNet data loaders
    # ------------------------

    base_batch_size = 512  # for teacher logits computation
    train_loader, val_loader = create_imagenet_dataloaders(train_batch_size=512, val_batch_size=512)

    # Use up to 10k samples (or fewer if dataset smaller)
    n_train_sub =  10000
    n_val_sub =  10000

    # Build subsets without materializing everything on GPU
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    n_train_total = len(train_dataset)
    n_val_total = len(val_dataset)

    n_train_sub = min(n_train_sub, n_train_total)
    n_val_sub = min(n_val_sub, n_val_total)

    # Choose random subset indices (shuffle to avoid ordering bias)
    rng = np.random.default_rng(seed=0)

    train_indices = rng.choice(len(train_dataset), size=n_train_sub, replace=False)
    val_indices   = rng.choice(len(val_dataset),   size=n_val_sub, replace=False)


    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"Using {len(train_subset)} train and {len(val_subset)} val samples "
          f"for TPV logit-noise experiment on ImageNet.")

    # ------------------------
    # Model list (diverse ImageNet architectures)
    # ------------------------
    MODEL_SPECS = [
        
        ("resnet50", models.resnet50),
        ("resnet18", models.resnet18),
        ("wide_resnet50_2", models.wide_resnet50_2),
        ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0),
        ("efficientnet_b0", models.efficientnet_b0),
        ("mnasnet1_0", models.mnasnet1_0),
        ("convnext_tiny", models.convnext_tiny),
    ]
    model_name_list = [m[0] for m in MODEL_SPECS]

    # ------------------------
    # Experiment config
    # ------------------------
    dataset_name = "imagenet"

    noise_std_list = [0.01] 
    # R = 10
    # max_epochs_noisy = 10
    R = 5
    max_epochs_noisy = 10

    lr_noisy = 1e-6 
    momentum = 0.9
    wd = 0.0
    opt = "sgd"
    proximity_lambda = 0 #0.001
    train_mse_thres = [0.0]  # per sigma
    force_train=True  # If True, ignore existing results and retrain
    accumulate_gradients = False  # If True: full-batch GD, if False: mini-batch GD

    train_batch_size = 64
    eval_batch_size = 64 

    fname = f"results/tpv_logit_noise_{dataset_name}.pkl" # bs (256,256)


    n_models = len(model_name_list)
    n_sigmas = len(noise_std_list)

    # ------------------------
    # Load existing results if available
    # ------------------------
    if os.path.exists(fname) and not force_train:
        print(f"\n=== Loading existing results from {fname} ===")
        with open(fname, "rb") as f:
            existing_results = pkl.load(f)
        
        # Extract existing data
        existing_model_names = existing_results.get("model_name_list", [])
        empirical_TPV_val = existing_results.get("empirical_TPV_val", np.zeros((n_models, n_sigmas)))
        empirical_TPV_train = existing_results.get("empirical_TPV_train", np.zeros((n_models, n_sigmas)))
        mse_train_noisy_targets_mean = existing_results.get("mse_train_noisy_targets_mean", np.zeros((n_models, n_sigmas)))
        mse_train_noisy_targets_std = existing_results.get("mse_train_noisy_targets_std", np.zeros((n_models, n_sigmas)))
        mse_train_clean_logits_mean = existing_results.get("mse_train_clean_logits_mean", np.zeros((n_models, n_sigmas)))
        mse_train_clean_logits_std = existing_results.get("mse_train_clean_logits_std", np.zeros((n_models, n_sigmas)))
        mse_val_clean_logits_mean = existing_results.get("mse_val_clean_logits_mean", np.zeros((n_models, n_sigmas)))
        mse_val_clean_logits_std = existing_results.get("mse_val_clean_logits_std", np.zeros((n_models, n_sigmas)))
        ce_val_clean_labels_mean = existing_results.get("ce_val_clean_labels_mean", np.zeros((n_models, n_sigmas)))
        ce_val_clean_labels_std = existing_results.get("ce_val_clean_labels_std", np.zeros((n_models, n_sigmas)))
        baseline_val_accuracy = existing_results.get("baseline_val_accuracy", np.zeros(n_models))
        baseline_mse_train_clean_logits = existing_results.get("baseline_mse_train_clean_logits", np.zeros(n_models))
        baseline_mse_val_clean_logits = existing_results.get("baseline_mse_val_clean_logits", np.zeros(n_models))
        total_loss_lol = existing_results.get("total_loss_lol", [[None for _ in range(n_sigmas)] for _ in range(n_models)])
        
        print(f"Loaded results for models: {existing_model_names}")
    else:
        print(f"\n=== No existing results found. Starting fresh. ===")
        existing_model_names = []
        empirical_TPV_val = np.zeros((n_models, n_sigmas))
        empirical_TPV_train = np.zeros((n_models, n_sigmas))
        mse_train_noisy_targets_mean = np.zeros((n_models, n_sigmas))
        mse_train_noisy_targets_std = np.zeros((n_models, n_sigmas))
        mse_train_clean_logits_mean = np.zeros((n_models, n_sigmas))
        mse_train_clean_logits_std = np.zeros((n_models, n_sigmas))
        mse_val_clean_logits_mean = np.zeros((n_models, n_sigmas))
        mse_val_clean_logits_std = np.zeros((n_models, n_sigmas))
        ce_val_clean_labels_mean = np.zeros((n_models, n_sigmas))
        ce_val_clean_labels_std = np.zeros((n_models, n_sigmas))
        baseline_val_accuracy = np.zeros(n_models)
        baseline_mse_train_clean_logits = np.zeros(n_models)
        baseline_mse_val_clean_logits = np.zeros(n_models)

        total_loss_lol = [[None for _ in range(n_sigmas)] for _ in range(n_models)]
    # ------------------------
    # Run experiment
    # ------------------------
    for mi, (model_name, model_ctor) in enumerate(MODEL_SPECS):
        # Check if model already processed
        if model_name in existing_model_names:
            print(f"\n=== Model {model_name} already processed. Skipping. ===")
            continue
            
        print(f"\n=== Model {model_name} ===")

        # 1) Load clean reference model (pretrained on ImageNet)
        print("  Loading pretrained clean reference model...")
        model_clean = model_ctor(pretrained=True).to(device)

        # Number of classes from model
        # For standard torchvision ImageNet models this is 1000
        num_classes = model_clean.fc.out_features if hasattr(model_clean, "fc") else 1000

        # 2) Compute teacher logits + labels for train and val subsets (on CPU)
        print("  Computing teacher logits for train subset...")
        teacher_train_logits, labels_train, baseline_mse_train, _ = compute_teacher_logits_and_labels(
            model_clean, train_subset, batch_size=base_batch_size, num_classes=num_classes
        )
        baseline_mse_train_clean_logits[mi] = baseline_mse_train

        print("  Computing teacher logits and baseline accuracy for val subset...")
        teacher_val_logits, labels_val, baseline_mse_val, baseline_val_acc = compute_teacher_logits_and_labels(
            model_clean, val_subset, batch_size=base_batch_size, num_classes=num_classes
        )
        baseline_mse_val_clean_logits[mi] = baseline_mse_val
        baseline_val_accuracy[mi] = baseline_val_acc

        print(f"  Baseline Val Accuracy (clean labels): {baseline_val_accuracy[mi] * 100:.2f}%")
        print(f"  Baseline MSE train loss (clean logits): {baseline_mse_train_clean_logits[mi]:.6f}")
        print(f"  Baseline MSE val   loss (clean logits): {baseline_mse_val_clean_logits[mi]:.6f}")

        base_state_dict = copy.deepcopy(model_clean.state_dict())

        # 3) For each noise std, compute empirical TPV and losses
        for si, sigma in enumerate(noise_std_list):
            print(f"    noise_std = {sigma:.5f}")

            stats = estimate_empirical_TPV_logit_noise(
                model_ctor=model_ctor,
                model_name=model_name,
                noise_std=sigma,
                R=R,
                train_subset=train_subset,
                val_subset=val_subset,
                teacher_train_logits=teacher_train_logits,
                teacher_val_logits=teacher_val_logits,
                labels_train=labels_train,
                labels_val=labels_val,
                base_state_dict=base_state_dict,
                max_epochs_noisy=max_epochs_noisy,
                train_mse_thres=train_mse_thres[si],
                lr_noisy=lr_noisy,
                wd=wd,
                momentum=momentum,
                opt=opt,
                proximity_lambda=proximity_lambda,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                accumulate_gradients=accumulate_gradients,
            )

            total_loss_lol[mi][si] = stats["total_loss_lol"]
            empirical_TPV_val[mi, si] = stats["empirical_TPV_val"]
            empirical_TPV_train[mi, si] = stats["empirical_TPV_train"]

            mse_train_noisy_targets_mean[mi, si] = stats["mse_train_noisy_targets_mean"]
            mse_train_noisy_targets_std[mi, si] = stats["mse_train_noisy_targets_std"]

            mse_train_clean_logits_mean[mi, si] = stats["mse_train_clean_logits_mean"]
            mse_train_clean_logits_std[mi, si] = stats["mse_train_clean_logits_std"]

            mse_val_clean_logits_mean[mi, si] = stats["mse_val_clean_logits_mean"]
            mse_val_clean_logits_std[mi, si] = stats["mse_val_clean_logits_std"]

            ce_val_clean_labels_mean[mi, si] = stats["ce_val_clean_labels_mean"]
            ce_val_clean_labels_std[mi, si] = stats["ce_val_clean_labels_std"]

        print(f"      Empirical TPV (Val):   {empirical_TPV_val[mi, :]}")
        print(f"      Empirical TPV (Train): {empirical_TPV_train[mi, :]}")
        print(f"      Val MSE vs clean logits (mean): {mse_val_clean_logits_mean[mi, :]}")
        print(f"      Val CE vs clean labels (mean): {ce_val_clean_labels_mean[mi, :]}")

        # Free model + tensors (teacher_logits stay on CPU; free explicitly if RAM an issue)
        del model_clean
        torch.cuda.empty_cache()

        # Save intermediate results after each model
        results = {
            "model_name_list": model_name_list,
            "noise_std_list": noise_std_list,
            "empirical_TPV_val": empirical_TPV_val,
            "empirical_TPV_train": empirical_TPV_train,
            "mse_train_noisy_targets_mean": mse_train_noisy_targets_mean,
            "mse_train_noisy_targets_std": mse_train_noisy_targets_std,
            "mse_train_clean_logits_mean": mse_train_clean_logits_mean,
            "mse_train_clean_logits_std": mse_train_clean_logits_std,
            "mse_val_clean_logits_mean": mse_val_clean_logits_mean,
            "mse_val_clean_logits_std": mse_val_clean_logits_std,
            "ce_val_clean_labels_mean": ce_val_clean_labels_mean,
            "ce_val_clean_labels_std": ce_val_clean_labels_std,
            "baseline_val_accuracy": baseline_val_accuracy,
            "baseline_mse_train_clean_logits": baseline_mse_train_clean_logits,
            "baseline_mse_val_clean_logits": baseline_mse_val_clean_logits,
            "R": R,
            "n_train": len(train_subset),
            "n_val": len(val_subset),
            "total_loss_lol": total_loss_lol,
        }
        with open(fname, "wb") as f:
            pkl.dump(results, f)
        print(f"  Saved intermediate results to: {fname}")

    # ------------------------
    # Save results
    # ------------------------
    results = {
        "model_name_list": model_name_list,
        "noise_std_list": noise_std_list,
        "empirical_TPV_val": empirical_TPV_val,
        "empirical_TPV_train": empirical_TPV_train,
        "mse_train_noisy_targets_mean": mse_train_noisy_targets_mean,
        "mse_train_noisy_targets_std": mse_train_noisy_targets_std,
        "mse_train_clean_logits_mean": mse_train_clean_logits_mean,
        "mse_train_clean_logits_std": mse_train_clean_logits_std,
        "mse_val_clean_logits_mean": mse_val_clean_logits_mean,
        "mse_val_clean_logits_std": mse_val_clean_logits_std,
        "ce_val_clean_labels_mean": ce_val_clean_labels_mean,
        "ce_val_clean_labels_std": ce_val_clean_labels_std,
        "baseline_val_accuracy": baseline_val_accuracy,
        "baseline_mse_train_clean_logits": baseline_mse_train_clean_logits,
        "baseline_mse_val_clean_logits": baseline_mse_val_clean_logits,
        "R": R,
        "n_train": len(train_subset),
        "n_val": len(val_subset),
        "total_loss_lol": total_loss_lol,
    }

    with open(fname, "wb") as f:
        pkl.dump(results, f)

    print("\nSaved results to:", fname)
    print("\nEmpirical TPV (Val):\n", empirical_TPV_val)
    print("\nEmpirical TPV (Train):\n", empirical_TPV_train)


    # ------------------------
    # Load results
    # ------------------------
    results = existing_results = pkl.load(open(fname, "rb"))

    existing_model_names = existing_results.get("model_name_list")
    empirical_TPV_val = existing_results.get("empirical_TPV_val")
    empirical_TPV_train = existing_results.get("empirical_TPV_train")
    mse_train_noisy_targets_mean = existing_results.get("mse_train_noisy_targets_mean")
    mse_train_noisy_targets_std = existing_results.get("mse_train_noisy_targets_std")
    mse_train_clean_logits_mean = existing_results.get("mse_train_clean_logits_mean")
    mse_train_clean_logits_std = existing_results.get("mse_train_clean_logits_std")
    mse_val_clean_logits_mean = existing_results.get("mse_val_clean_logits_mean")
    mse_val_clean_logits_std = existing_results.get("mse_val_clean_logits_std")
    ce_val_clean_labels_mean = existing_results.get("ce_val_clean_labels_mean")
    ce_val_clean_labels_std = existing_results.get("ce_val_clean_labels_std")
    baseline_val_accuracy = existing_results.get("baseline_val_accuracy")
    baseline_mse_train_clean_logits = existing_results.get("baseline_mse_train_clean_logits")
    baseline_mse_val_clean_logits = existing_results.get("baseline_mse_val_clean_logits")
    total_loss_lol = existing_results.get('total_loss_lol', [])

    model_name_list = results["model_name_list"]
    model_name_list_short = [f"{i}" for i in range(len(model_name_list))]
    noise_std_list = results["noise_std_list"]

    # ------------------------
    # Single plot: TPV (x) vs Val Accuracy (y)
    # ------------------------
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 14
    TITLE_FONTSIZE = 18
    LEGEND_FONTSIZE = 14

    sigma_idx_for_plot = 0
    sigma_val = noise_std_list[sigma_idx_for_plot]

    x_train_tpv = empirical_TPV_train[:, sigma_idx_for_plot]
    x_val_tpv = empirical_TPV_val[:, sigma_idx_for_plot]
    y_val_acc = baseline_val_accuracy  # vector: val accuracy of clean reference model

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # ------------------------
    # Scatter: Train TPV and Val TPV
    # ------------------------
    plt.scatter(
        x_train_tpv,
        y_val_acc,
        marker="^",
        label="Train TPV",
        s=80,
    )

    plt.scatter(
        x_val_tpv,
        y_val_acc,
        marker="s",
        label="Val TPV",
        s=40,
    )

    # ------------------------
    # Linear trend lines (best-fit straight line)
    # ------------------------
    def add_trend_line(x, y, label):
        # Only add if we have at least 2 distinct x values
        if np.unique(x).size > 1:
            coef = np.polyfit(x, y, 1)  # slope, intercept
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = np.polyval(coef, x_line)
            plt.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=1.5,
                label=label,
            )

    add_trend_line(x_train_tpv, y_val_acc, "Train TPV trend")
    add_trend_line(x_val_tpv, y_val_acc, "Val TPV trend")

    # ------------------------
    # Text annotations with auto-flip at right boundary
    # ------------------------
    # Horizontal offset in data coordinates
    text_offset = 0.025 * (x_train_tpv.max() - x_train_tpv.min() + 1e-8)
    text_offset2 = 0.02 * (y_val_acc.max() - y_val_acc.min() + 1e-8)
    for i, name in enumerate(model_name_list):
        x = x_train_tpv[i]
        y = y_val_acc[i]
        # Proposed right-side annotation position in data coordinates
        x_right = x + text_offset
        y_right = y

        # Convert to display (pixel) coordinates
        x_disp, y_disp = ax.transData.transform((x_right, y_right))

        # Right boundary of the axes in display coordinates
        x_max_disp = ax.transAxes.transform((1, 0))[0]

        print(name, x_disp , x_max_disp)

        if x_disp > x_max_disp-50:
            # If it would go outside, put text on the left
            plt.text(
                x - text_offset,
                y,
                f"{name}",
                fontsize=8,
                ha="right",
                va="center",
            )
        else:
            # Otherwise keep it on the right as usual
            plt.text(
                x + text_offset,
                y-text_offset2,
                f"{name}",
                fontsize=8,
                ha="left",
                va="center",
            )

    # ------------------------
    # Labels, legend, save
    # ------------------------
    plt.xlabel("Empirical TPV", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Validation Accuracy", fontsize=LABEL_FONTSIZE)
    plt.title(f"TPV vs Accuracy (sigma = {sigma_val})", fontsize=TITLE_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE - 3)
    plt.yticks(fontsize=TICK_FONTSIZE)

    legend = plt.legend(fontsize=LEGEND_FONTSIZE - 2, loc="upper right")
    legend.get_frame().set_alpha(0)

    plt.tight_layout()
    plot_path = f"results/plots/{dataset_name}.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", plot_path)