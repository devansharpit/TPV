import math
import numpy as np
import pickle as pkl
import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
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

os.makedirs('./results/', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)


# -------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(
        description="TPV CIFAR label noise experiment."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Which dataset: cifar10 (CIFAR-10) or cifar100 (CIFAR-100)",
    )
    return parser.parse_args()

# ----------------------------
# Training (MSE on logits)
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


def train_full_batch_mse(
    model,
    X,
    Y_targets,
    max_epochs=1000,
    train_mse_thres=0.001,
    lr=1e-3,
    batch_size=None,
    wd=0.0,
    momentum=0.9,
    opt="sgd",
    print_stats=False,
    ref_state_dict=None,
    proximity_lambda=0.0,
    dataloader_seed=None,
):
    """
    Train model to regress its logits to Y_targets using MSE loss.
    Training stops when MSE goes below train_mse_thres or max_epochs is reached.

    X:          (N, C, H, W)
    Y_targets:  (N, K) - target logits (teacher logits + Gaussian noise)

    If batch_size is None, uses full-batch (implemented via gradient
    accumulation over a fixed accumulation batch size).
    
    ref_state_dict: If provided, adds proximity penalty: proximity_lambda * ||w - w_ref||^2
    proximity_lambda: Weight of the proximity penalty term.
    dataloader_seed: If provided, uses a fixed seed for DataLoader shuffling (deterministic across runs).
    
    Returns: (final_loss, epochs_trained)
    """
    # model.train()
    model.eval()
    if opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    criterion = nn.MSELoss()
    n_samples = X.shape[0]

    use_minibatch = batch_size is not None and batch_size < n_samples
    accum_batch_size = 500  # for full-batch via accumulation

    if use_minibatch:
        # Use a dedicated generator with fixed seed for deterministic shuffling across runs
        if dataloader_seed is not None:
            loader_gen = torch.Generator()
            loader_gen.manual_seed(dataloader_seed)
        else:
            loader_gen = None
        
        dataset = torch.utils.data.TensorDataset(X, Y_targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,#True,
            generator=loader_gen,
        )
    else:
        dataset = torch.utils.data.TensorDataset(X, Y_targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=accum_batch_size, shuffle=False)

    epochs_trained = 0
    total_loss_list = []

    # Check training MSE initially
    with torch.no_grad():
        logits = model(X)
        current_loss = criterion(logits, Y_targets).item()
    # Append current loss to list
    total_loss_list.append(current_loss)
    print(f"[MSE] Initial train MSE (noisy targets)={current_loss}")

    pbar = tqdm(range(max_epochs), desc="Training")
    for epoch in pbar:
        epochs_trained = epoch + 1
        if use_minibatch:
            for batch_X, batch_Y in dataloader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_Y)
                # Add proximity penalty if reference parameters provided
                if ref_state_dict is not None and proximity_lambda > 0:
                    prox_penalty = compute_proximity_penalty(model, ref_state_dict)
                    loss = loss + proximity_lambda * prox_penalty
                if not torch.isfinite(loss):
                    print(f"[train_full_batch_mse] Non-finite loss at epoch {epoch}: {loss.item()}")
                    return None, epochs_trained, total_loss_list
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        else:
            optimizer.zero_grad()
            total_loss = 0.0
            for batch_X, batch_Y in dataloader:
                logits = model(batch_X)
                batch_weight = batch_X.shape[0] / n_samples
                loss = criterion(logits, batch_Y) * batch_weight
                if not torch.isfinite(loss):
                    print(f"[train_full_batch_mse] Non-finite loss at epoch {epoch}: {loss.item()}")
                    return None, epochs_trained, total_loss_list
                loss.backward()
                total_loss += loss.item()
            # Add proximity penalty if reference parameters provided (once per epoch for full-batch)
            if ref_state_dict is not None and proximity_lambda > 0:
                prox_penalty = compute_proximity_penalty(model, ref_state_dict)
                (proximity_lambda * prox_penalty).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Check training MSE after each epoch
        model.eval()
        with torch.no_grad():
            logits = model(X)
            current_loss = criterion(logits, Y_targets).item()
        model.eval()
        
        # Append current loss to list
        total_loss_list.append(current_loss)
        pbar.set_postfix({"MSE": f"{current_loss}"})


        # Stop if training MSE is below threshold
        if current_loss < train_mse_thres:
            break

    print(f"[MSE] Final train MSE (noisy targets)={current_loss}")
    if total_loss_list[-1]>total_loss_list[0]:
        print(f"[WARNING] Loss exploded: {total_loss_list[-1]}>{total_loss_list[0]}")
    final_loss = None
    return final_loss, epochs_trained, total_loss_list


def eval_full_batch_ce(model, X, y, batch_size=256):
    """
    Evaluate cross-entropy loss on (X, y) using minibatches.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            logits = model(batch_X)
            total_loss += criterion(logits, batch_y).item()
    return total_loss / n_samples


def eval_full_batch_mse(model, X, target_logits, batch_size=256):
    """
    Evaluate MSE loss between model logits and target_logits on (X, target_logits).
    """
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    n_samples = X.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_Y = target_logits[i:i+batch_size]
            logits = model(batch_X)
            total_loss += criterion(logits, batch_Y).item()
    return total_loss / n_samples


# ----------------------------
# Empirical TPV under logit Gaussian noise
# ----------------------------
def estimate_empirical_TPV_logit_noise(
    model_name,
    noise_std,
    R,
    X_train,
    y_train,
    X_test,
    y_test,
    base_state_dict,
    f_star_train_logits,
    f_star_test_logits,
    max_epochs_noisy=1000,
    train_mse_thres=0.001,
    lr_noisy=1e-3,
    batch_size=None,
    momentum=0.9,
    wd=0.0,
    opt="sgd",
    proximity_lambda=0.0,
):
    """
    For fixed model and Gaussian logit noise std = noise_std:

    - f_star is the clean reference model logits.
    - For each run r:
        * Sample noisy logit targets:
              y_noisy_logits = f_star_train_logits + eps,
              eps ~ N(0, noise_std^2 I).
        * Initialize model at w* (base_state_dict).
        * Fine-tune using MSE on logits with optional proximity penalty to w*.
        * Record logits on train/test and various losses.

    We define vector-output TPV as:
        TPV = E_{runs,x}[ || z_hat(x) - z_star(x) ||_2^2 ].
    
    IMPORTANT: We make SGD / DataLoader behavior deterministic across runs
    and let ONLY label noise differ between runs. This is done via:
        - global seeding at the top of the file (torch / np / random)
        - a dedicated DataLoader generator with a fixed seed each run
        - label noise sampling using the global torch RNG (no reseeding)
    """
    preds_runs_train = []
    preds_runs_test = []

    mse_train_noisy_targets_list = []
    mse_train_clean_logits_list = []
    mse_test_clean_logits_list = []

    ce_train_clean_labels_list = []
    ce_test_clean_labels_list = []

    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # Dedicated seed for DataLoader shuffling; we will use this
    # SAME seed each run so minibatch order is identical.
    LOADER_SEED = 12345
    
    total_loss_lol = []

    for r in range(R):
        # Start from clean reference model weights
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            model_name,
            pretrained=False,
            verbose=False,
        ).to(device)
        model.load_state_dict(base_state_dict)

        # --- LABEL NOISE (DIFFERENT ACROSS RUNS) ---
        # Use global torch RNG (seeded once at top). We DO NOT reseed here.
        eps_train = torch.randn_like(f_star_train_logits) * noise_std
        y_noisy_logits = f_star_train_logits + eps_train

        # --- TRAINING (IDENTICAL ACROSS RUNS except for noisy targets) ---
        # Fine-tune with MSE on logits until threshold is reached
        # Optionally includes proximity penalty to reference model parameters
        # Use LOADER_SEED to ensure deterministic minibatch order across runs
        _, epochs_trained, total_loss_list = train_full_batch_mse(
            model,
            X_train,
            y_noisy_logits,
            max_epochs=max_epochs_noisy,
            train_mse_thres=train_mse_thres,
            lr=lr_noisy,
            batch_size=batch_size,
            wd=wd,
            momentum=momentum,
            opt=opt,
            print_stats=True,
            ref_state_dict=base_state_dict,
            proximity_lambda=proximity_lambda,
            dataloader_seed=LOADER_SEED,  # <- deterministic minibatch order
        )
        total_loss_lol.append(total_loss_list)

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits_train = model(X_train)  # (n_train, K)
            logits_test = model(X_test)    # (n_test, K)

        preds_runs_train.append(logits_train.cpu().numpy())
        preds_runs_test.append(logits_test.cpu().numpy())

        # Losses wrt noisy and clean logits
        mse_train_noisy_targets_list.append(
            mse_criterion(logits_train, y_noisy_logits).item()
        )
        mse_train_clean_logits_list.append(
            mse_criterion(logits_train, f_star_train_logits).item()
        )
        mse_test_clean_logits_list.append(
            mse_criterion(logits_test, f_star_test_logits).item()
        )

        # CE losses wrt clean labels
        ce_train_clean_labels_list.append(
            ce_criterion(logits_train, y_train).item()
        )
        ce_test_clean_labels_list.append(
            ce_criterion(logits_test, y_test).item()
        )

    preds_runs_train = np.stack(preds_runs_train, axis=0)  # (R, n_train, K)
    preds_runs_test = np.stack(preds_runs_test, axis=0)    # (R, n_test, K)

    f_star_train_np = f_star_train_logits.cpu().numpy()    # (n_train, K)
    f_star_test_np = f_star_test_logits.cpu().numpy()      # (n_test, K)

    # Empirical TPV: mean squared deviation from clean logits over runs and samples
    diffs_train = preds_runs_train - f_star_train_np[None, :, :]  # (R, n_train, K)
    diffs_test = preds_runs_test - f_star_test_np[None, :, :]     # (R, n_test, K)

    sqnorm_train = np.sum(diffs_train ** 2, axis=-1)  # (R, n_train)
    sqnorm_test = np.sum(diffs_test ** 2, axis=-1)    # (R, n_test)

    empirical_TPV_train = np.mean(sqnorm_train)
    empirical_TPV_test = np.mean(sqnorm_test)

    # Aggregate losses
    mse_train_noisy_targets_mean = np.mean(mse_train_noisy_targets_list)
    mse_train_noisy_targets_std = np.std(mse_train_noisy_targets_list)

    mse_train_clean_logits_mean = np.mean(mse_train_clean_logits_list)
    mse_train_clean_logits_std = np.std(mse_train_clean_logits_list)

    mse_test_clean_logits_mean = np.mean(mse_test_clean_logits_list)
    mse_test_clean_logits_std = np.std(mse_test_clean_logits_list)

    ce_train_clean_labels_mean = np.mean(ce_train_clean_labels_list)
    ce_train_clean_labels_std = np.std(ce_train_clean_labels_list)

    ce_test_clean_labels_mean = np.mean(ce_test_clean_labels_list)
    ce_test_clean_labels_std = np.std(ce_test_clean_labels_list)

    return dict(
        empirical_TPV_train=empirical_TPV_train,
        empirical_TPV_test=empirical_TPV_test,
        mse_train_noisy_targets_mean=mse_train_noisy_targets_mean,
        mse_train_noisy_targets_std=mse_train_noisy_targets_std,
        mse_train_clean_logits_mean=mse_train_clean_logits_mean,
        mse_train_clean_logits_std=mse_train_clean_logits_std,
        mse_test_clean_logits_mean=mse_test_clean_logits_mean,
        mse_test_clean_logits_std=mse_test_clean_logits_std,
        ce_train_clean_labels_mean=ce_train_clean_labels_mean,
        ce_train_clean_labels_std=ce_train_clean_labels_std,
        ce_test_clean_labels_mean=ce_test_clean_labels_mean,
        ce_test_clean_labels_std=ce_test_clean_labels_std,
        total_loss_lol=total_loss_lol,
    )


# ----------------------------
# CIFAR data (small subsets)
# ----------------------------
args = get_args()
dataset =  args.dataset  #["cifar10", "cifar100"][0]  # "cifar10" or "cifar100"
np.random.seed(0)
n_train_sub = 4000
n_test_sub = 4000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

if dataset=="cifar10":
    num_classes = 10
    data_cls = torchvision.datasets.CIFAR10
    train_mse_thres = [0] #[0.003, 0.01]  # training stops when MSE goes below this threshold
else:
    num_classes = 100
    data_cls = torchvision.datasets.CIFAR100
    train_mse_thres = [0] # [0.03, 0.1]  # training stops when MSE goes below this threshold
train_dataset = data_cls(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = data_cls(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)



train_indices = np.random.choice(len(train_dataset), n_train_sub, replace=False)
test_indices = np.random.choice(len(test_dataset), n_test_sub, replace=False)

X_train_list = []
y_train_list = []
for i in train_indices:
    x, y = train_dataset[i]
    X_train_list.append(x)
    y_train_list.append(y)
X_train = torch.stack(X_train_list, dim=0).to(device)   # (n_train_sub, 3, 32, 32)
y_train = torch.tensor(y_train_list, dtype=torch.long, device=device)

X_test_list = []
y_test_list = []
for i in test_indices:
    x, y = test_dataset[i]
    X_test_list.append(x)
    y_test_list.append(y)
X_test = torch.stack(X_test_list, dim=0).to(device)
y_test = torch.tensor(y_test_list, dtype=torch.long, device=device)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"Using {n_train} train and {n_test} test samples for TPV logit-noise experiment.")


# ----------------------------
# Main experiment config
# ----------------------------



model_name_list = [
    f"{dataset}_mobilenetv2_x0_5",
    f"{dataset}_mobilenetv2_x0_75",
    f"{dataset}_mobilenetv2_x1_0",
    f"{dataset}_mobilenetv2_x1_4",
]

# Standard deviations of Gaussian noise on logits (teacher outputs)
noise_std_list = [0.1, ] 
R = 20                  # Monte Carlo runs per (model, noise_std)
max_epochs_noisy = 10  # maximum epochs 

lr_noisy = 1e-4
batch_size = 256 # Full-batch training using gradient accumulation if None
momentum = 0.9
wd=0.0
opt = "sgd"
proximity_lambda = 0 # 0.001  # weight for proximity penalty ||w - w_ref||^2 (0 = no penalty)
fname = f"results/tpv_logit_noise_{dataset}.pkl" # target noise is the only source of randomness



n_models = len(model_name_list)
n_sigmas = len(noise_std_list)

empirical_TPV_test = np.zeros((n_models, n_sigmas))
empirical_TPV_train = np.zeros((n_models, n_sigmas))

mse_train_noisy_targets_mean = np.zeros((n_models, n_sigmas))
mse_train_noisy_targets_std = np.zeros((n_models, n_sigmas))

mse_train_clean_logits_mean = np.zeros((n_models, n_sigmas))
mse_train_clean_logits_std = np.zeros((n_models, n_sigmas))

mse_test_clean_logits_mean = np.zeros((n_models, n_sigmas))
mse_test_clean_logits_std = np.zeros((n_models, n_sigmas))

ce_train_clean_labels_mean = np.zeros((n_models, n_sigmas))
ce_train_clean_labels_std = np.zeros((n_models, n_sigmas))

ce_test_clean_labels_mean = np.zeros((n_models, n_sigmas))
ce_test_clean_labels_std = np.zeros((n_models, n_sigmas))

# Baseline clean-model CE on labels and MSE on teacher logits
baseline_ce_test_clean_labels = np.zeros(n_models)
baseline_ce_train_clean_labels = np.zeros(n_models)
baseline_mse_test_clean_logits = np.zeros(n_models)   # should be ~0
baseline_mse_train_clean_logits = np.zeros(n_models)  # should be ~0

# Initialize total_loss_lol as a list of lists
total_loss_lol = [[None for _ in range(n_sigmas)] for _ in range(n_models)]

# ----------------------------
# Run experiment
# ----------------------------
for mi, model_name in enumerate(model_name_list):
    print(f"\n=== Model {model_name} ===")

    # 1) Load clean reference model (pretrained on CIFAR)
    print("  Loading pretrained clean reference model...")
    model_clean = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        verbose=False,
    ).to(device)
    model_clean.eval()

    # Clean logits on train and test: z*(x)
    with torch.no_grad():
        f_star_train_logits = model_clean(X_train)  # (n_train, K)
        f_star_test_logits = model_clean(X_test)    # (n_test, K)

    # Baseline CE on clean labels
    baseline_ce_train_clean_labels[mi] = eval_full_batch_ce(model_clean, X_train, y_train)
    baseline_ce_test_clean_labels[mi] = eval_full_batch_ce(model_clean, X_test, y_test)

    # Baseline MSE between teacher and itself (should be 0 up to numerical noise)
    baseline_mse_train_clean_logits[mi] = eval_full_batch_mse(
        model_clean, X_train, f_star_train_logits
    )
    baseline_mse_test_clean_logits[mi] = eval_full_batch_mse(
        model_clean, X_test, f_star_test_logits
    )

    print(f"  Baseline CE train loss (clean labels): {baseline_ce_train_clean_labels[mi]:.4f}")
    print(f"  Baseline CE test  loss (clean labels): {baseline_ce_test_clean_labels[mi]:.4f}")
    print(f"  Baseline MSE train loss (clean logits): {baseline_mse_train_clean_logits[mi]:.6f}")
    print(f"  Baseline MSE test  loss (clean logits): {baseline_mse_test_clean_logits[mi]:.6f}")

    base_state_dict = copy.deepcopy(model_clean.state_dict())

    # 2) For each noise std, compute empirical TPV and losses
    for si, sigma in enumerate(noise_std_list):
        print(f"    noise_std = {sigma:.3f}")

        stats = estimate_empirical_TPV_logit_noise(
            model_name=model_name,
            noise_std=sigma,
            R=R,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            base_state_dict=base_state_dict,
            f_star_train_logits=f_star_train_logits,
            f_star_test_logits=f_star_test_logits,
            max_epochs_noisy=max_epochs_noisy,
            train_mse_thres=train_mse_thres[si],
            lr_noisy=lr_noisy,
            batch_size=batch_size,
            wd=wd,
            momentum=momentum,
            opt=opt,
            proximity_lambda=proximity_lambda,
        )

        total_loss_lol[mi][si] = stats["total_loss_lol"]
        empirical_TPV_test[mi, si] = stats["empirical_TPV_test"]
        empirical_TPV_train[mi, si] = stats["empirical_TPV_train"]

        mse_train_noisy_targets_mean[mi, si] = stats["mse_train_noisy_targets_mean"]
        mse_train_noisy_targets_std[mi, si] = stats["mse_train_noisy_targets_std"]

        mse_train_clean_logits_mean[mi, si] = stats["mse_train_clean_logits_mean"]
        mse_train_clean_logits_std[mi, si] = stats["mse_train_clean_logits_std"]

        mse_test_clean_logits_mean[mi, si] = stats["mse_test_clean_logits_mean"]
        mse_test_clean_logits_std[mi, si] = stats["mse_test_clean_logits_std"]

        ce_train_clean_labels_mean[mi, si] = stats["ce_train_clean_labels_mean"]
        ce_train_clean_labels_std[mi, si] = stats["ce_train_clean_labels_std"]

        ce_test_clean_labels_mean[mi, si] = stats["ce_test_clean_labels_mean"]
        ce_test_clean_labels_std[mi, si] = stats["ce_test_clean_labels_std"]

    print(f"      Empirical TPV (Test):  {empirical_TPV_test[mi, :]}")
    print(f"      Empirical TPV (Train): {empirical_TPV_train[mi, :]}")
    print(f"      Train MSE vs clean logits (mean): {mse_train_clean_logits_mean[mi, :]}")
    print(f"      Test MSE vs clean logits (mean): {mse_test_clean_logits_mean[mi, :]}")
    print(f"      Test CE vs clean labels (mean): {ce_test_clean_labels_mean[mi, :]}")


# ----------------------------
# Save results
# ----------------------------
results = {
    "model_name_list": model_name_list,
    "noise_std_list": noise_std_list,
    "empirical_TPV_test": empirical_TPV_test,
    "empirical_TPV_train": empirical_TPV_train,
    "mse_train_noisy_targets_mean": mse_train_noisy_targets_mean,
    "mse_train_noisy_targets_std": mse_train_noisy_targets_std,
    "mse_train_clean_logits_mean": mse_train_clean_logits_mean,
    "mse_train_clean_logits_std": mse_train_clean_logits_std,
    "mse_test_clean_logits_mean": mse_test_clean_logits_mean,
    "mse_test_clean_logits_std": mse_test_clean_logits_std,
    "ce_train_clean_labels_mean": ce_train_clean_labels_mean,
    "ce_train_clean_labels_std": ce_train_clean_labels_std,
    "ce_test_clean_labels_mean": ce_test_clean_labels_mean,
    "ce_test_clean_labels_std": ce_test_clean_labels_std,
    "baseline_ce_train_clean_labels": baseline_ce_train_clean_labels,
    "baseline_ce_test_clean_labels": baseline_ce_test_clean_labels,
    "baseline_mse_train_clean_logits": baseline_mse_train_clean_logits,
    "baseline_mse_test_clean_logits": baseline_mse_test_clean_logits,
    "R": R,
    "n_train": n_train,
    "n_test": n_test,
    "total_loss_lol": total_loss_lol,
}


with open(fname, "wb") as f:
    pkl.dump(results, f)

print("\nSaved results to:", fname)
print("\nEmpirical TPV (Test):\n", empirical_TPV_test)
print("\nEmpirical TPV (Train):\n", empirical_TPV_train)











####################################
# Plotting
####################################

results_path = 'results'
dataset = args.dataset # 'cifar10'

# ----------------------------
# Reload and plotting
# ----------------------------
results = pkl.load(open(fname, "rb"))
model_name_list = results["model_name_list"]
noise_std_list = results["noise_std_list"]
empirical_TPV_test = results["empirical_TPV_test"]
empirical_TPV_train = results["empirical_TPV_train"]
mse_test_clean_logits_mean = results["mse_test_clean_logits_mean"]
mse_test_clean_logits_std = results["mse_test_clean_logits_std"]
ce_test_clean_labels_mean = results["ce_test_clean_labels_mean"]
ce_test_clean_labels_std = results["ce_test_clean_labels_std"]
baseline_ce_test_clean_labels = results["baseline_ce_test_clean_labels"]

model_name_list_short = ['x0.5', 'x0.75', 'x1.0', 'x1.4']

LABEL_FONTSIZE = 30
TICK_FONTSIZE = 25
TITLE_FONTSIZE = 25
LEGEND_FONTSIZE = 20

x = np.arange(len(model_name_list))

# 1) TPV vs model for first noise_std
sigma_idx_for_model = 0 #####################################################################
sigma_val = noise_std_list[sigma_idx_for_model]

plt.figure(figsize=(6, 4))
plt.plot(
    x,
    empirical_TPV_test[:, sigma_idx_for_model],
    marker="s",
    color='tab:blue',
    label="Empirical TPV (Test)",
)
plt.plot(
    x,
    empirical_TPV_train[:, sigma_idx_for_model],
    marker="^",
    color='tab:green',
    label="Empirical TPV (Train)",
)
plt.xlabel("Models", fontsize=LABEL_FONTSIZE)
plt.ylabel("TPV", fontsize=LABEL_FONTSIZE)
plt.title(f"TPV vs Model (sigma = {sigma_val})", fontsize=TITLE_FONTSIZE)
plt.xticks(x, model_name_list_short, rotation=20, fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE-5)
legend = plt.legend(fontsize=LEGEND_FONTSIZE)
legend.get_frame().set_alpha(0)
plt.tight_layout()
plt.savefig(f"{results_path}/plots/{dataset}_tpv_vs_model_sigma-{sigma_val}.pdf", bbox_inches="tight")
plt.show()

# 2) CE test loss and TPV vs model for first noise_std
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Models", fontsize=LABEL_FONTSIZE)
ax1.set_ylabel("CE Test Loss", color="black", fontsize=LABEL_FONTSIZE)
line1 = ax1.errorbar(
    x,
    ce_test_clean_labels_mean[:, sigma_idx_for_model],
    yerr=ce_test_clean_labels_std[:, sigma_idx_for_model],
    color='tab:orange',
    marker="o",
    linestyle="-",
    label="CE Test Loss (Noisy Model)",
    capsize=5,
    capthick=2,
)
line3 = ax1.plot(
    x,
    baseline_ce_test_clean_labels,
    color='tab:brown',
    marker="o",
    linestyle="-",
    label="CE Test Loss (Ref. Model)",
)
ax1.tick_params(axis="y", labelcolor="black", labelsize=TICK_FONTSIZE)
ax1.tick_params(axis="x", labelsize=TICK_FONTSIZE)
ax1.set_xticks(x)
ax1.set_xticklabels(model_name_list_short, rotation=20)

# Second y-axis for empirical TPV (sum over logits)
ax2 = ax1.twinx()
ax2.set_ylabel("Empirical TPV (test)", color="tab:blue", fontsize=LABEL_FONTSIZE)
line2 = ax2.plot(
    x,
    empirical_TPV_test[:, sigma_idx_for_model],
    marker="^",
    linestyle="-",
    label="Empirical TPV (test)",
    linewidth=2,
    markersize=8,
)
ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=TICK_FONTSIZE)

plt.title(f"CE Test Loss and TPV vs Model (sigma = {sigma_val})", fontsize=TITLE_FONTSIZE)

lines = line3 + [line1]  + line2
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, loc="upper right", fontsize=LEGEND_FONTSIZE)
legend.get_frame().set_alpha(0)

plt.tight_layout()
plt.savefig(f"{results_path}/plots/{dataset}_ce_and_tpv_vs_model_sigma-{sigma_val}.pdf", bbox_inches="tight")
plt.show()