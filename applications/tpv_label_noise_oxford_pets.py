import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle as pkl
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import OxfordIIITPet
from tpv.label_noise import LabelTPV

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

NUM_CLASSES_PETS = 37


# ----------------------------
# Label corruption helpers
# ----------------------------
class LabelCorruptedDataset(Dataset):
    """
    Wraps an existing dataset and flips a fraction of the target labels
    to a uniformly random *different* class.
    """
    def __init__(self, base_dataset, corrupted_targets):
        self.base_dataset = base_dataset
        self.corrupted_targets = corrupted_targets  # list / array of ints

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]          # ignore original label
        return img, self.corrupted_targets[idx]


def corrupt_dataset_labels(
    dataset,
    num_classes: int = NUM_CLASSES_PETS,
    corrupt_fraction: float = 0.10,
    seed: int = 42,
):
    """
    Return a `LabelCorruptedDataset` where `corrupt_fraction` of the
    labels are flipped to a uniformly random *different* class.

    Parameters
    ----------
    dataset : torchvision-style dataset with (image, label) indexing.
    num_classes : total number of classes (37 for Oxford Pets).
    corrupt_fraction : proportion of samples whose labels are flipped.
    seed : RNG seed for reproducibility.

    Returns
    -------
    LabelCorruptedDataset
        A thin wrapper that serves corrupted labels while keeping the
        original images and transforms intact.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    n_corrupt = int(n * corrupt_fraction)

    # Collect original labels
    original_labels = []
    for i in range(n):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        original_labels.append(int(label))
    corrupted_labels = list(original_labels)

    # Choose which indices to corrupt
    corrupt_indices = rng.choice(n, size=n_corrupt, replace=False)

    for idx in corrupt_indices:
        old_label = corrupted_labels[idx]
        new_label = None
        while new_label is None or new_label == old_label:
            # Pick a random class that differs from the original
            new_label = rng.integers(0, num_classes)
        corrupted_labels[idx] = int(new_label)

    n_actually_flipped = sum(
        c != o for c, o in zip(corrupted_labels, original_labels)
    )
    print(f"Label corruption: flipped {n_actually_flipped}/{n} labels "
          f"({n_actually_flipped / n * 100:.1f}%)")

    return LabelCorruptedDataset(dataset, corrupted_labels)


# ----------------------------
# Oxford Pets dataloader
# ----------------------------
def create_oxford_pets_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    corrupt_fraction: float = 0.10,
    corrupt_seed: int = 42,
):
    """
    Download (if needed) and return train/val DataLoaders for Oxford-IIIT Pet.

    Oxford Pets has a standard trainval / test split via torchvision.
    We use:
        split="trainval"  ->  3,680 images across 37 classes  (our "train")
        split="test"      ->  3,669 images across 37 classes  (our "val")

    Images are resized to 224x224 with standard ImageNet normalisation so that
    pretrained torchvision weights transfer directly.

    An additional corrupted-label version of the trainval set is also
    returned, where `corrupt_fraction` of the labels are flipped.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = OxfordIIITPet(
        root=data_root, split="trainval", download=True, transform=val_transform
    )
    val_dataset = OxfordIIITPet(
        root=data_root, split="test", download=True, transform=val_transform
    )

    # Build a corrupted-label version of the trainval set
    corrupted_train_dataset = corrupt_dataset_labels(
        train_dataset,
        num_classes=NUM_CLASSES_PETS,
        corrupt_fraction=corrupt_fraction,
        seed=corrupt_seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    corrupted_train_loader = DataLoader(
        corrupted_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )

    print(f"Oxford Pets — train: {len(train_dataset)} samples, "
          f"corrupted train: {len(corrupted_train_dataset)} samples, "
          f"val: {len(val_dataset)} samples, classes: {NUM_CLASSES_PETS}")
    return train_loader, corrupted_train_loader, val_loader


# ----------------------------
# Head replacement
# ----------------------------
def replace_classification_head(model, model_name: str,
                                num_classes: int = NUM_CLASSES_PETS,
                                dropout_p: float = 0.0):
    """
    Replace the final classification layer of a torchvision model with a
    new Linear(in_features, num_classes) layer, initialised randomly.

    If ``dropout_p > 0``, a ``nn.Dropout(p=dropout_p)`` is inserted
    immediately before the new Linear so that dropout acts on the
    backbone features.  For head slots that are already an
    ``nn.Sequential`` (e.g. torchvision's EfficientNet classifier),
    the Dropout is inserted in-place before the last Linear.  For head
    slots that are a bare ``nn.Linear`` (e.g. ResNet's ``.fc``,
    ConvNeXt's ``.head``), the slot is replaced with
    ``nn.Sequential(Dropout, Linear)``.

    Handles the most common torchvision head patterns:
        .fc          -- ResNet, WideResNet, ShuffleNet, MNASNet
        .classifier  -- EfficientNet, ConvNeXt (Sequential, last element)
        .head        -- (fallback for ViT-style if added later)
    """
    # Detect the device the model currently lives on so that the newly
    # created head ends up on the same device (avoids CPU/CUDA mismatch).
    try:
        target_device = next(model.parameters()).device
    except StopIteration:
        target_device = None

    def _new_linear_with_optional_dropout(in_features):
        """Return either a bare Linear (dropout_p==0) or a Sequential
        (Dropout, Linear) so the structure of the slot stays simple."""
        lin = nn.Linear(in_features, num_classes)
        if dropout_p > 0:
            return nn.Sequential(nn.Dropout(p=dropout_p), lin)
        return lin

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = _new_linear_with_optional_dropout(in_features)

    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = _new_linear_with_optional_dropout(clf.in_features)
        elif isinstance(clf, nn.Sequential):
            # Find the last Linear layer in the Sequential
            last_linear_idx = None
            for i, layer in enumerate(clf):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx is not None:
                in_features = clf[last_linear_idx].in_features
                # Replace the last Linear with a fresh one.
                clf[last_linear_idx] = nn.Linear(in_features, num_classes)
                # If requested, slot a Dropout immediately before it.
                if dropout_p > 0:
                    # Replace in-place: build a new Sequential of the
                    # modules before the last Linear + Dropout + new Linear
                    # + anything after (typically nothing).
                    modules_before = list(clf.children())[:last_linear_idx]
                    modules_after  = list(clf.children())[last_linear_idx+1:]
                    new_last_linear = clf[last_linear_idx]  # freshly replaced
                    model.classifier = nn.Sequential(
                        *modules_before,
                        nn.Dropout(p=dropout_p),
                        new_last_linear,
                        *modules_after,
                    )
            else:
                raise ValueError(f"No Linear layer found in classifier for {model_name}")

    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = _new_linear_with_optional_dropout(in_features)

    else:
        raise ValueError(f"Cannot identify classification head for model: {model_name}. "
                         f"Add a custom case to replace_classification_head().")

    # Move the newly created head layer to the same device as the rest of the model.
    if target_device is not None:
        model.to(target_device)

    return model


def get_num_classes_from_head(model, model_name: str) -> int:
    """
    Read out_features from whichever head pattern the model uses.
    Call this AFTER replace_classification_head() to confirm the swap.

    Handles both plain Linear heads and Sequential(Dropout, ..., Linear)
    heads that arise from ``replace_classification_head(..., dropout_p>0)``.
    """
    def _last_linear_out_features(seq):
        for layer in reversed(seq):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        return None

    if hasattr(model, "fc"):
        if isinstance(model.fc, nn.Linear):
            return model.fc.out_features
        if isinstance(model.fc, nn.Sequential):
            out = _last_linear_out_features(model.fc)
            if out is not None:
                return out
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            return clf.out_features
        if isinstance(clf, nn.Sequential):
            out = _last_linear_out_features(clf)
            if out is not None:
                return out
    if hasattr(model, "head"):
        if isinstance(model.head, nn.Linear):
            return model.head.out_features
        if isinstance(model.head, nn.Sequential):
            out = _last_linear_out_features(model.head)
            if out is not None:
                return out
    raise ValueError(f"Cannot read num_classes from head for {model_name}")


# ----------------------------
# Helpers: identify head parameters
# ----------------------------
def get_head_module(model, model_name: str = ""):
    """
    Return a reference to the final Linear layer (the classification head)
    of a torchvision model.  Raises if it cannot find one.

    Handles both plain Linear heads and Sequential(Dropout, ..., Linear)
    heads that arise from ``replace_classification_head(..., dropout_p>0)``.
    """
    def _last_linear(seq):
        for layer in reversed(seq):
            if isinstance(layer, nn.Linear):
                return layer
        return None

    if hasattr(model, "fc"):
        if isinstance(model.fc, nn.Linear):
            return model.fc
        if isinstance(model.fc, nn.Sequential):
            lin = _last_linear(model.fc)
            if lin is not None:
                return lin
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            return clf
        if isinstance(clf, nn.Sequential):
            lin = _last_linear(clf)
            if lin is not None:
                return lin
    if hasattr(model, "head"):
        if isinstance(model.head, nn.Linear):
            return model.head
        if isinstance(model.head, nn.Sequential):
            lin = _last_linear(model.head)
            if lin is not None:
                return lin
    raise ValueError(f"Cannot identify head module for model: {model_name}")


def _set_requires_grad_head_only(model, model_name: str = ""):
    """Freeze everything except the final Linear head."""
    for param in model.parameters():
        param.requires_grad_(False)
    head = get_head_module(model, model_name)
    for param in head.parameters():
        param.requires_grad_(True)


def _set_requires_grad_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)



# ----------------------------
# Two-phase fine-tuning to obtain w*
#   Phase 1: head-only (linear probe) to warm up the randomly-initialised head
#   Phase 2: full-network with a smaller backbone LR
#
# The two-phase protocol avoids scrambling the pretrained backbone features
# with large gradients coming from a random head at the start of training.
# ----------------------------
def finetune_head(
    model,
    train_loader,
    num_epochs: int = 10,               # Phase 1: head-only epochs
    lr: float = 1e-3,                   # Phase 1 / head LR in Phase 2
    momentum: float = 0.9,
    wd: float = 1e-4,
    labelsmooth: float = 0.0,
    # --- Phase 2 (full-network) controls ---
    backbone_finetune_epochs: int = 0,  # 0 disables Phase 2 (head-only behaviour)
    backbone_lr: float = 1e-4,          # smaller LR for backbone in Phase 2
    opt: str = "adamw",                  # optimizer for Phase 2 (default: AdamW)
    model_name: str = "",
    device=device,
):
    """
    Obtain w* via a two-phase fine-tuning protocol:
        Phase 1: Linear probe.  Freeze the backbone, train ONLY the
                 classification head with learning rate ``lr`` for
                 ``num_epochs`` epochs.  This brings the randomly
                 initialised head to a reasonable solution without
                 disturbing the pretrained backbone features.
        Phase 2: Full fine-tuning.  Unfreeze the entire network and
                 train for ``backbone_finetune_epochs`` epochs with
                 per-parameter-group LRs:
                     - backbone: ``backbone_lr`` (typically 10-100x
                       smaller than ``lr``),
                     - head:     ``lr``.
                 This lets the backbone adapt slightly to Oxford Pets
                 so that subsequent noisy-logit TPV estimation actually
                 probes the full parameter subspace, not just the head.

    If ``backbone_finetune_epochs == 0``, only Phase 1 is run (original
    head-only behaviour).

    Regularization knobs:
      ``wd``          : SGD weight_decay applied to the optimized parameters.
      ``labelsmooth`` : label-smoothing epsilon in cross-entropy loss.
      dropout         : already baked into the head via
                        ``replace_classification_head(..., dropout_p=...)``;
                        this function does not set it directly.

    Returns the model in eval() mode with all parameters set to
    requires_grad=True so that the subsequent noisy-logit fine-tuning
    operates over the full parameter vector.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=labelsmooth)

    # --------------------------------------------------------------
    # Phase 1: head-only linear probe
    # --------------------------------------------------------------
    _set_requires_grad_head_only(model, model_name)
    if opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd, momentum=momentum,
        )
    elif opt == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd,
        )

    model.train()
    for epoch in range(num_epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for images, labels in tqdm(
            train_loader,
            desc=f"  [Phase1 head] epoch {epoch+1}/{num_epochs}",
            leave=False,
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)
        print(f"  [Phase1 head] epoch {epoch+1:02d} | "
              f"loss={total_loss/total_samples:.4f} | "
              f"acc={total_correct/total_samples*100:.2f}%")

    # --------------------------------------------------------------
    # Phase 2: full-network fine-tuning with differential LRs
    # --------------------------------------------------------------
    if backbone_finetune_epochs > 0:
        _set_requires_grad_all(model)

        head = get_head_module(model, model_name)
        head_param_ids = {id(p) for p in head.parameters()}
        backbone_params = [
            p for p in model.parameters() if id(p) not in head_param_ids
        ]
        head_params = list(head.parameters())

        if opt == "sgd":
            optimizer = optim.SGD(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params,     "lr": lr},
                ],
                weight_decay=wd,
                momentum=momentum,
            )
        elif opt == "adamw":
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params,     "lr": lr},
                ],
                weight_decay=wd,
            )

        model.train()
        for epoch in range(backbone_finetune_epochs):
            # divide lr by 10 when epoch hits 50% of backbone_finetune_epochs to avoid overshooting
            if epoch == int(0.5 * backbone_finetune_epochs):
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for images, labels in tqdm(
                train_loader,
                desc=f"  [Phase2 full] epoch {epoch+1}/{backbone_finetune_epochs}",
                leave=False,
            ):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += images.size(0)
            print(f"  [Phase2 full] epoch {epoch+1:02d} | "
                  f"loss={total_loss/total_samples:.4f} | "
                  f"acc={total_correct/total_samples*100:.2f}% | "
                  f"lr_bb={backbone_lr:.1e} lr_hd={lr:.1e}")

    # --------------------------------------------------------------
    # Return: unfreeze everything, set to eval() mode.
    # The noisy-logit fine-tuning that follows optimises all parameters.
    # --------------------------------------------------------------
    _set_requires_grad_all(model)
    model.eval()
    return model


# ----------------------------
# LabelTPV helpers
# ----------------------------

def subset_to_tensor(subset, batch_size=64, num_workers=4):
    """Load a Subset into a CPU float tensor (images only)."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    return torch.cat([imgs for imgs, _ in loader], dim=0)


def make_train_fn(max_epochs, lr, opt, batch_size):
    """Return a train_fn(model, X_train, y_noisy) -> None for LabelTPV."""
    def train_fn(model, X_train, y_noisy):
        model.eval()
        if opt == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_noisy)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(12345),
        )
        for _ in range(max_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                criterion(model(bx.to(device)), by.to(device)).backward()
                optimizer.step()
    return train_fn


def compute_teacher_logits_and_labels(model, subset, batch_size, num_classes):
    """Return (teacher_logits, labels, baseline_mse, baseline_acc, baseline_ce).

    ``baseline_ce`` is the cross-entropy of the teacher predictions
    against the clean labels on this subset, with no label smoothing.
    It is used by the caller to overlay training-loss markers on TPV
    plots (Sec.~5.3 / Sec.~6.2 style)."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    N = len(subset)
    teacher_logits = torch.empty((N, num_classes), dtype=torch.float32)
    labels = torch.empty((N,), dtype=torch.long)
    mse_criterion = nn.MSELoss(reduction="sum")
    ce_criterion = nn.CrossEntropyLoss(reduction="sum")
    total_mse, total_correct, total_ce, total_samples = 0.0, 0, 0.0, 0

    model.eval()
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            bsz = images.size(0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            teacher_logits[offset:offset + bsz].copy_(logits.cpu())
            labels[offset:offset + bsz].copy_(targets.cpu())
            offset += bsz
            total_mse += mse_criterion(logits, logits).item()
            total_correct += (preds == targets).sum().item()
            total_ce += ce_criterion(logits, targets).item()
            total_samples += bsz

    baseline_mse = total_mse / max(total_samples, 1)
    baseline_acc = total_correct / max(total_samples, 1)
    baseline_ce  = total_ce  / max(total_samples, 1)
    return teacher_logits, labels, baseline_mse, baseline_acc, baseline_ce



# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    # ------------------------
    # Experiment config
    # ------------------------
    # We sweep (backbone x joint regularization config) cells, mirroring the
    # CIFAR MLP multi-regularizer experiment (Sec 6.2 in the paper).  Unlike
    # that experiment, each "architecture" here is an ImageNet-pretrained
    # backbone, and the downstream task is Oxford-IIIT Pets with 10% label
    # noise injected into the downstream training set.
    #
    # Each joint configuration specifies a triple
    #     (wd, dropout_p, labelsmooth)
    # sampled once from fixed candidate grids; the same N_CONFIGS triples
    # are applied to every backbone.  Each cell produces one w* via a
    # two-phase fine-tune (head-only then head+backbone) on the noisy
    # Oxford Pets training set, after which we estimate TPV by a clean
    # noisy-logit fine-tune (no regularization -- see Sec 4.1).

    dataset_name   = "oxford_pets_multireg"
    DATA_ROOT      = "./data"
    CORRUPT_FRAC   = 0.1     # 10% label noise on downstream train set
    CORRUPT_SEED   = 42

    base_batch_size   = 128
    train_batch_size  = 64
    eval_batch_size   = 64

    # Two-phase fine-tune: head-only warmup, then joint head+backbone update.
    HEAD_FINETUNE_EPOCHS     = 1      # Phase 1 (head-only)
    BACKBONE_FINETUNE_EPOCHS = 10       # Phase 2 (head + backbone joint)
    HEAD_LR                  = 1e-3    # Phase 1 LR / head LR in Phase 2
    BACKBONE_LR              = 1e-3    # Phase 2 backbone LR (100x smaller)
    FINETUNE_OPT = "adamw"

    # This is part of the label-noise TPV estimation algorithm (Sec 4.1).
    noise_std_list   = [0.1]
    R                = 5
    max_epochs_noisy = 5
    lr_noisy         = 1e-6
    opt              = "adamw"

    # Joint regularization configurations.
    # Candidate grids -- same spirit as the CIFAR MLP experiment.
    WD_CANDIDATES          = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    DROPOUT_CANDIDATES     = [0.0, 0.1,  0.2, ]
    LABELSMOOTH_CANDIDATES = [0.0, 0.05, 0.1, ]

    N_CONFIGS   = 5
    CONFIG_SEED = 0

    _cfg_rng = np.random.default_rng(CONFIG_SEED)
    REG_CONFIGS = []
    for ci in range(N_CONFIGS):
        REG_CONFIGS.append(dict(
            config_id   = ci,
            wd          = float(_cfg_rng.choice(WD_CANDIDATES)),
            dropout     = float(_cfg_rng.choice(DROPOUT_CANDIDATES)),
            labelsmooth = float(_cfg_rng.choice(LABELSMOOTH_CANDIDATES)),
        ))

    def config_short_label(c):
        wd = c["wd"]; dp = c["dropout"]; ls = c["labelsmooth"]
        wd_s = f"wd={wd:.0e}" if wd > 0 else "wd=0"
        return f"C{c['config_id']}: {wd_s}, dropout={dp:g}, labelsmooth={ls:g}"

    print("\nSampled regularization configurations:")
    for c in REG_CONFIGS:
        print(f"  {config_short_label(c)}")

    # ------------------------
    # Oxford Pets data loaders
    # trainval: 3,680 images | test: 3,669 images | 37 classes
    # ``train_loader`` below uses labels with CORRUPT_FRAC of them randomized;
    # ``clean_train_loader`` (discarded) preserves original labels.  The
    # val_loader always uses clean labels.
    # ------------------------
    _, train_loader, val_loader = create_oxford_pets_dataloaders(
        data_root=DATA_ROOT,
        batch_size=base_batch_size,
        corrupt_fraction=CORRUPT_FRAC,
        corrupt_seed=CORRUPT_SEED,
    )

    # Expose raw train/val datasets for TPV estimation (subset wrapping).
    # For TPV estimation we use the SAME (corrupted) training set the
    # model actually fit to -- this is what w* corresponds to.
    train_dataset = train_loader.dataset
    val_dataset   = val_loader.dataset

    rng = np.random.default_rng(seed=0)
    train_indices = rng.choice(len(train_dataset), size=len(train_dataset), replace=False)
    val_indices   = rng.choice(len(val_dataset),   size=len(val_dataset),   replace=False)
    train_subset  = Subset(train_dataset, train_indices)
    val_subset    = Subset(val_dataset,   val_indices)
    print(f"Using {len(train_subset)} train and {len(val_subset)} val samples.")

    # ------------------------
    # Backbones to sweep
    # ------------------------
    MODEL_SPECS = [
        ("resnet50",        models.resnet50,        models.ResNet50_Weights.IMAGENET1K_V1),
        ("resnet18",        models.resnet18,        models.ResNet18_Weights.IMAGENET1K_V1),
        ("efficientnet_b0", models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        ("convnext_tiny",   models.convnext_tiny,   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    ]
    model_name_list = [m[0] for m in MODEL_SPECS]
    n_models  = len(MODEL_SPECS)
    n_configs = len(REG_CONFIGS)
    n_sigmas  = len(noise_std_list)

    fname = (f"results/tpv_logit_noise_{dataset_name}_"
             f"n{N_CONFIGS}_seed{CONFIG_SEED}.pkl")

    # ------------------------
    # Result containers: (n_models, n_configs, n_sigmas) for TPV-like
    # quantities; (n_models, n_configs) for per-cell scalar quantities.
    # ------------------------
    force_train = True  # if True, re-run even if cached results exist.

    if (not force_train) and os.path.exists(fname):
        print(f"\n=== Loading existing results from {fname} ===")
        with open(fname, "rb") as f:
            prior = pkl.load(f)
        existing_cells = set(prior.get("completed_cells", []))
        empirical_TPV_val       = prior["empirical_TPV_val"]
        empirical_TPV_train     = prior["empirical_TPV_train"]
        baseline_val_accuracy   = prior["baseline_val_accuracy"]
        baseline_ce_train_ref   = prior["baseline_ce_train_ref"]
        baseline_ce_val_ref     = prior["baseline_ce_val_ref"]
    else:
        existing_cells = set()
        empirical_TPV_val       = np.zeros((n_models, n_configs, n_sigmas))
        empirical_TPV_train     = np.zeros((n_models, n_configs, n_sigmas))
        baseline_val_accuracy   = np.zeros((n_models, n_configs))
        baseline_ce_train_ref   = np.zeros((n_models, n_configs))
        baseline_ce_val_ref     = np.zeros((n_models, n_configs))

    label_tpv = LabelTPV(device=device, seed=GLOBAL_SEED)

    # Pre-load train/val subsets as CPU tensors once (LabelTPV expects tensors).
    print("Pre-loading train subset into CPU tensor...")
    X_train_t = subset_to_tensor(train_subset, batch_size=eval_batch_size)
    print("Pre-loading val subset into CPU tensor...")
    X_test_t  = subset_to_tensor(val_subset,   batch_size=eval_batch_size)
    print(f"  X_train_t: {tuple(X_train_t.shape)}  X_test_t: {tuple(X_test_t.shape)}")

    # ------------------------
    # Main (backbone x config) loop
    # ------------------------
    for mi, (model_name, model_ctor, weights) in enumerate(MODEL_SPECS):
        for ci, rc in enumerate(REG_CONFIGS):
            cell_key = f"{model_name}__C{rc['config_id']}"
            if cell_key in existing_cells:
                print(f"\n=== Cell {cell_key} already processed. Skipping. ===")
                continue

            print(f"\n=== Model {model_name} | {config_short_label(rc)} ===")

            # 1) Load ImageNet pretrained backbone
            print("  Loading ImageNet pretrained model...")
            model_clean = model_ctor(weights=weights).to(device)

            # 2) Replace head with a fresh 37-class Linear
            #    (with optional dropout slotted in via replace_classification_head).
            replace_classification_head(
                model_clean, model_name,
                num_classes=NUM_CLASSES_PETS,
                dropout_p=rc["dropout"],
            )
            nc = get_num_classes_from_head(model_clean, model_name)
            assert nc == NUM_CLASSES_PETS, (
                f"Head replacement failed: got {nc} classes")
            print(f"  Head replaced: {nc} classes, head-dropout p={rc['dropout']:g}")

            # 3) Two-phase fine-tune on Oxford Pets (corrupted train set)
            #    Phase 1: head-only warmup
            #    Phase 2: head + backbone joint fine-tune (smaller backbone LR)
            print(f"  Fine-tuning w* -- Phase1(head)={HEAD_FINETUNE_EPOCHS} ep, "
                  f"Phase2(head+bb)={BACKBONE_FINETUNE_EPOCHS} ep | "
                  f"wd={rc['wd']:.0e}, ls={rc['labelsmooth']:g}")
            model_clean = finetune_head(
                model_clean, train_loader,
                num_epochs=HEAD_FINETUNE_EPOCHS,
                lr=HEAD_LR, momentum=0.9,
                wd=rc["wd"],
                labelsmooth=rc["labelsmooth"],
                backbone_finetune_epochs=BACKBONE_FINETUNE_EPOCHS,
                backbone_lr=BACKBONE_LR,
                opt=FINETUNE_OPT,
                model_name=model_name,
            )

            # 4) Baseline train/val CE (clean labels) at w*
            print("  Computing baseline CE for train subset...")
            _, _, _, train_acc, ce_tr_ref = \
                compute_teacher_logits_and_labels(
                    model_clean, train_subset,
                    batch_size=base_batch_size, num_classes=nc,
                )
            print("  Computing baseline CE for val subset...")
            _, _, _, val_acc, ce_val_ref = \
                compute_teacher_logits_and_labels(
                    model_clean, val_subset,
                    batch_size=base_batch_size, num_classes=nc,
                )
            baseline_val_accuracy[mi, ci] = val_acc
            baseline_ce_train_ref[mi, ci] = ce_tr_ref   # train CE (overlay)
            baseline_ce_val_ref[mi, ci]   = ce_val_ref  # test/val CE (y-axis)
            print(f"  Val acc: {val_acc*100:.2f}%  |  "
                  f"train CE (noisy labels): {ce_tr_ref:.4f}  |  "
                  f"val CE (clean labels): {ce_val_ref:.4f}")

            base_state_dict = copy.deepcopy(model_clean.state_dict())

            # 5) TPV estimation -- per sigma
            def make_model_ctor(ctor=model_ctor, name=model_name,
                                dp=rc["dropout"]):
                # Reconstruct an empty model with the SAME head structure
                # (dropout + Linear) so load_state_dict(base_state_dict) works.
                def _ctor():
                    m = ctor(weights=None)
                    replace_classification_head(
                        m, name, num_classes=NUM_CLASSES_PETS, dropout_p=dp)
                    return m
                return _ctor

            for si, sigma in enumerate(noise_std_list):
                print(f"    TPV noise_std = {sigma:.5f}")

                train_fn = make_train_fn(
                    max_epochs=max_epochs_noisy,
                    lr=lr_noisy,
                    opt=opt,
                    batch_size=train_batch_size,
                )
                stats = label_tpv.compute_tpv(
                    model_factory=make_model_ctor(),
                    base_state_dict=base_state_dict,
                    X_train=X_train_t,
                    X_test=X_test_t,
                    noise_std=sigma,
                    R=R,
                    train_fn=train_fn,
                    batch_size=eval_batch_size,
                )
                empirical_TPV_val[mi, ci, si]   = stats["empirical_TPV_test"]
                empirical_TPV_train[mi, ci, si] = stats["empirical_TPV_train"]

            del model_clean
            torch.cuda.empty_cache()

            # Incremental save after each cell.
            existing_cells.add(cell_key)
            results_dict = {
                "model_name_list":   model_name_list,
                "reg_configs":       REG_CONFIGS,
                "noise_std_list":    noise_std_list,
                "empirical_TPV_val": empirical_TPV_val,
                "empirical_TPV_train": empirical_TPV_train,
                "baseline_val_accuracy": baseline_val_accuracy,
                "baseline_ce_train_ref": baseline_ce_train_ref,
                "baseline_ce_val_ref":   baseline_ce_val_ref,
                "R":                 R,
                "n_train":           len(train_subset),
                "n_val":             len(val_subset),
                "completed_cells":   sorted(existing_cells),
                "corrupt_fraction":  CORRUPT_FRAC,
            }
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, "wb") as f:
                pkl.dump(results_dict, f)
            print(f"  Saved intermediate results ({len(existing_cells)}/"
                  f"{n_models*n_configs} cells) to {fname}")

    print(f"\nAll cells done. Final results at {fname}")

    results_dict = pkl.load(open(fname, "rb"))
    empirical_TPV_val     = results_dict["empirical_TPV_val"]
    empirical_TPV_train   = results_dict["empirical_TPV_train"]
    baseline_val_accuracy = results_dict["baseline_val_accuracy"]
    baseline_ce_train_ref = results_dict["baseline_ce_train_ref"]
    baseline_ce_val_ref   = results_dict["baseline_ce_val_ref"]

    # ------------------------
    # Spearman correlation: TPV (train) vs Val accuracy across all (arch, config) pairs
    # ------------------------
    from scipy.stats import spearmanr
    sigma_idx=0
    tpv_flat = empirical_TPV_train[:, :, sigma_idx].flatten()
    acc_flat = baseline_val_accuracy.flatten()
    rho, pval = spearmanr(tpv_flat, acc_flat)
    print(f"\nSpearman correlation (TPV_train vs Val accuracy): rho={rho:.4f}, p={pval:.4e}")
    print(f"  ({len(tpv_flat)} points: {n_models} architectures x {n_configs} configs)\n")

    # ------------------------
    # Plot -- same style as CIFAR MLP multi-reg plot.
    # X: empirical TPV (log).  Y-left (solid): val acc (clean labels).
    # Y-right (faded): train CE on the downstream (noisy) training set.
    # Color: backbone.  Marker: joint reg config.
    # ------------------------
    sigma_idx = 0
    sigma_val = noise_std_list[sigma_idx]

    pair_labels = list(model_name_list)
    pair_colors = plt.cm.tab10(np.linspace(0, 0.9, len(pair_labels)))
    color_map   = dict(zip(pair_labels, pair_colors))

    all_markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    marker_map  = {f"config_{c['config_id']}": all_markers[c['config_id'] % len(all_markers)]
                   for c in REG_CONFIGS}

    LABEL_FONTSIZE  = 16
    TICK_FONTSIZE   = 12
    LEGEND_FONTSIZE = 9

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax2 = ax.twinx()

    for mi, bb_name in enumerate(model_name_list):
        for ci, rc in enumerate(REG_CONFIGS):
            x  = empirical_TPV_train[mi, ci, sigma_idx]
            yL = baseline_val_accuracy[mi, ci]
            yR = baseline_ce_train_ref[mi, ci]   # train CE on noisy labels (faded)
            ax.scatter(x, yL,
                       color=color_map[bb_name],
                       marker=marker_map[f"config_{rc['config_id']}"],
                       s=120, zorder=3, linewidths=0.5, edgecolors="k")
            ax2.scatter(x, yR,
                        color=color_map[bb_name],
                        marker=marker_map[f"config_{rc['config_id']}"],
                        s=50, zorder=3, linewidths=0.2, edgecolors="k",
                        alpha=0.35)

    # Dashed lines connecting same-backbone points, sorted by TPV.
    for mi, bb_name in enumerate(model_name_list):
        xs_line, ys_line = [], []
        triples = [(empirical_TPV_train[mi, ci, sigma_idx],
                    baseline_val_accuracy[mi, ci])
                   for ci in range(n_configs)]
        triples.sort(key=lambda t: t[0])
        xs_line = [t[0] for t in triples]
        ys_line = [t[1] for t in triples]
        ax.plot(xs_line, ys_line, "--",
                color=color_map[bb_name], linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("TPV (train)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Val accuracy (solid markers)", fontsize=LABEL_FONTSIZE)

    ax2.set_ylabel("Train CE on noisy labels (faded markers)",
                   fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax2.tick_params(labelsize=TICK_FONTSIZE)

    legend_elements = (
        [Line2D([0], [0], color=color_map[bb], lw=0, marker="o",
                markersize=8, label=bb) for bb in pair_labels]
        + [Line2D([0], [0], color="k", lw=0,
                  marker=marker_map[f"config_{rc['config_id']}"],
                  markersize=8, label=config_short_label(rc))
           for rc in REG_CONFIGS]
    )
    ax.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE,
              loc="upper left", framealpha=0, bbox_to_anchor=(0, 0.5))
    plt.tight_layout()

    plot_path = (f"results/plots/{dataset_name}_tpv_"
                 f"n{N_CONFIGS}_seed{CONFIG_SEED}.pdf")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {plot_path}")

    # ------------------------
    # Summary table
    # ------------------------
    print(f"\n{'Backbone':<18} {'Config':<28} {'Val acc':>8} "
          f"{'Train CE':>9} {'Val CE':>8} {'TPV (tr)':>12}")
    print("-" * 90)
    for mi, bb_name in enumerate(model_name_list):
        for ci, rc in enumerate(REG_CONFIGS):
            print(f"{bb_name:<18} {config_short_label(rc):<28} "
                  f"{baseline_val_accuracy[mi, ci]*100:>7.2f}% "
                  f"{baseline_ce_train_ref[mi, ci]:>9.4f} "
                  f"{baseline_ce_val_ref[mi, ci]:>8.4f} "
                  f"{empirical_TPV_train[mi, ci, sigma_idx]:>12.4e}")
