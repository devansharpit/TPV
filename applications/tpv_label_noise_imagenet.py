import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from torchvision import models
from torch.utils.data import DataLoader, Subset

from tpv.label_noise import LabelTPV
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


# ----------------------------
# Main: ImageNet TPV experiment (memory efficient)
# ----------------------------
if __name__ == "__main__":
    # ------------------------
    # ImageNet data loaders
    # ------------------------


    base_batch_size = 256  # for teacher logits computation
    train_loader, val_loader = create_imagenet_dataloaders(batch_size=512, val_batch_size=512, use_augmentation=False)

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

    noise_std_list = [0.1]
    R = 5
    max_epochs_noisy = 5

    lr_noisy = 1e-8
    momentum = 0.9
    wd = 0.0
    opt = "adamw"
    force_train = True  # If True, ignore existing results and retrain

    train_batch_size = 64

    fname = f"results/tpv_logit_noise_{dataset_name}.pkl"


    n_models = len(model_name_list)
    n_sigmas = len(noise_std_list)

    # ------------------------
    # Load existing results if available
    # ------------------------
    if os.path.exists(fname) and not force_train:
        print(f"\n=== Loading existing results from {fname} ===")
        with open(fname, "rb") as f:
            existing_results = pkl.load(f)

        existing_model_names = existing_results.get("model_name_list", [])
        empirical_TPV_val = existing_results.get("empirical_TPV_val", np.zeros((n_models, n_sigmas)))
        empirical_TPV_train = existing_results.get("empirical_TPV_train", np.zeros((n_models, n_sigmas)))
        baseline_val_accuracy = existing_results.get("baseline_val_accuracy", np.zeros(n_models))
        baseline_mse_train_clean_logits = existing_results.get("baseline_mse_train_clean_logits", np.zeros(n_models))
        baseline_mse_val_clean_logits = existing_results.get("baseline_mse_val_clean_logits", np.zeros(n_models))

        print(f"Loaded results for models: {existing_model_names}")
    else:
        print(f"\n=== No existing results found. Starting fresh. ===")
        existing_model_names = []
        empirical_TPV_val = np.zeros((n_models, n_sigmas))
        empirical_TPV_train = np.zeros((n_models, n_sigmas))
        baseline_val_accuracy = np.zeros(n_models)
        baseline_mse_train_clean_logits = np.zeros(n_models)
        baseline_mse_val_clean_logits = np.zeros(n_models)
    # ------------------------
    # Run experiment
    # ------------------------
    def subset_to_tensor(subset, batch_size=512):
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        return torch.cat([imgs.to(device) for imgs, _ in loader], dim=0)

    label_tpv = LabelTPV(device=device, seed=GLOBAL_SEED)

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

        # 2) Compute baseline stats for train and val subsets
        print("  Computing baseline stats for train subset...")
        _, _, baseline_mse_train, _ = compute_teacher_logits_and_labels(
            model_clean, train_subset, batch_size=base_batch_size, num_classes=num_classes
        )
        baseline_mse_train_clean_logits[mi] = baseline_mse_train

        print("  Computing baseline stats and accuracy for val subset...")
        _, _, baseline_mse_val, baseline_val_acc = compute_teacher_logits_and_labels(
            model_clean, val_subset, batch_size=base_batch_size, num_classes=num_classes
        )
        baseline_mse_val_clean_logits[mi] = baseline_mse_val
        baseline_val_accuracy[mi] = baseline_val_acc

        print(f"  Baseline Val Accuracy (clean labels): {baseline_val_accuracy[mi] * 100:.2f}%")
        print(f"  Baseline MSE train loss (clean logits): {baseline_mse_train_clean_logits[mi]:.6f}")
        print(f"  Baseline MSE val   loss (clean logits): {baseline_mse_val_clean_logits[mi]:.6f}")

        base_state_dict = copy.deepcopy(model_clean.state_dict())

        # 3) Materialize subsets as GPU tensors for LabelTPV
        print("  Materializing train/val subsets as tensors...")
        X_train_tensor = subset_to_tensor(train_subset)
        X_test_tensor  = subset_to_tensor(val_subset)

        def model_factory():
            return model_ctor(pretrained=False)

        # 4) For each noise std, compute empirical TPV via LabelTPV
        for si, sigma in enumerate(noise_std_list):
            print(f"    noise_std = {sigma:.5f}")

            def train_fn(m, X, y_noisy):
                m.eval()
                if opt == "adamw":
                    optimizer = optim.AdamW(m.parameters(), lr=lr_noisy, weight_decay=wd)
                else:
                    optimizer = optim.SGD(m.parameters(), lr=lr_noisy, momentum=momentum, weight_decay=wd)
                criterion = nn.MSELoss()
                loader = DataLoader(
                    torch.utils.data.TensorDataset(X, y_noisy),
                    batch_size=train_batch_size,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(12345),
                    num_workers=0,
                )
                for _ in range(max_epochs_noisy):
                    for bx, by in loader:
                        optimizer.zero_grad()
                        criterion(m(bx), by).backward()
                        optimizer.step()

            stats = label_tpv.compute_tpv(
                model_factory=model_factory,
                base_state_dict=base_state_dict,
                X_train=X_train_tensor,
                X_test=X_test_tensor,
                noise_std=sigma,
                R=R,
                train_fn=train_fn,
                batch_size=base_batch_size,
            )

            empirical_TPV_train[mi, si] = stats["empirical_TPV_train"]
            empirical_TPV_val[mi, si]   = stats["empirical_TPV_test"]

        print(f"      Empirical TPV (Val):   {empirical_TPV_val[mi, :]}")
        print(f"      Empirical TPV (Train): {empirical_TPV_train[mi, :]}")

        # Free model and GPU tensors
        del model_clean, X_train_tensor, X_test_tensor
        torch.cuda.empty_cache()

        # Save intermediate results after each model
        results = {
            "model_name_list": model_name_list,
            "noise_std_list": noise_std_list,
            "empirical_TPV_val": empirical_TPV_val,
            "empirical_TPV_train": empirical_TPV_train,
            "baseline_val_accuracy": baseline_val_accuracy,
            "baseline_mse_train_clean_logits": baseline_mse_train_clean_logits,
            "baseline_mse_val_clean_logits": baseline_mse_val_clean_logits,
            "R": R,
            "n_train": len(train_subset),
            "n_val": len(val_subset),
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
        "baseline_val_accuracy": baseline_val_accuracy,
        "baseline_mse_train_clean_logits": baseline_mse_train_clean_logits,
        "baseline_mse_val_clean_logits": baseline_mse_val_clean_logits,
        "R": R,
        "n_train": len(train_subset),
        "n_val": len(val_subset),
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

    empirical_TPV_val = existing_results.get("empirical_TPV_val")
    empirical_TPV_train = existing_results.get("empirical_TPV_train")
    baseline_val_accuracy = existing_results.get("baseline_val_accuracy")

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
    plt.xticks(fontsize=TICK_FONTSIZE - 3)
    plt.yticks(fontsize=TICK_FONTSIZE)

    legend = plt.legend(fontsize=LEGEND_FONTSIZE - 2, loc="upper right")
    legend.get_frame().set_alpha(0)

    plt.tight_layout()
    plot_path = f"results/plots/{dataset_name}_tpv_vs_valacc.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", plot_path)
