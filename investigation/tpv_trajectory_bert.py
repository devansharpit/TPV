import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
TPV trajectory when training BERT-Small on AG News / TREC with BERT-Small
==========================================================

Pipeline:
  1. Fine-tune BERT-Small (prajjwal1/bert-small, 4L/512H, ~29M params) on a
     text classification task (AG News or TREC) with a standard recipe
     (AdamW, linear warmup+decay) with a 
     fraction of training set class labels corrupted randomly.
  2. Every TPV_EVERY epochs, snapshot the model and estimate empirical
     training-set TPV via noisy-logit fine-tuning on a fixed subset
     (no augmentation; pre-tokenized).
  3. Plot training accuracy, validation accuracy, and training-set TPV
     against epoch on a dual-axis figure.

Usage:
    python tpv_trajectory_bert.py --task ag_news --noise_ratio 0.20 --epochs 50 --lr 1e-4 --labelsmooth 0
    python tpv_trajectory_bert.py --task trec --noise_ratio 0.20 --epochs 50 --lr 1e-4 --labelsmooth 0

"""

import argparse
import copy
import pickle as pkl
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ----------------------------
# Argument parsing
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="TPV trajectory on AG News / TREC with BERT-Small."
    )
    # Task / model
    parser.add_argument("--task", type=str, default="ag_news",
                        choices=["ag_news", "trec"],
                        help="Text classification task.")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-small",
                        help="HuggingFace model identifier.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Tokenizer max length.")

    # Main training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--labelsmooth", type=float, default=0.0)

    # Label noise
    parser.add_argument("--noise_ratio", type=float, default=0.30,
                        help="Fraction of training labels to corrupt. "
                             "Suggested: 0.30 for AG News, 0.20 for TREC.")

    # TPV estimator
    parser.add_argument("--tpv_every", type=int, default=1,
                        help="Compute TPV every N epochs (also at epoch 0).")
    parser.add_argument("--tpv_subset_size", type=int, default=3000,
                        help="Number of training samples for TPV estimation.")
    parser.add_argument("--R", type=int, default=3,
                        help="Number of noisy fine-tuning runs per TPV estimate.")
    parser.add_argument("--max_epochs_noisy", type=int, default=3,
                        help="Fine-tuning epochs per noisy run.")
    parser.add_argument("--lr_noisy", type=float, default=1e-5,
                        help="Learning rate for noisy fine-tuning.")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="sigma multiplier for adaptive logit noise.")
    parser.add_argument("--tpv_batch_size", type=int, default=64)
    parser.add_argument("--no_adaptive_noise_scaling", action="store_true")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


args = get_args()
ADAPTIVE_NOISE_SCALING = not args.no_adaptive_noise_scaling


# ----------------------------
# Reproducibility & device
# ----------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("./results/", exist_ok=True)
os.makedirs("./results/plots/", exist_ok=True)


# ----------------------------
# Data loading & tokenization
# ----------------------------
TASK_CONFIG = {
    "ag_news": {
        "hf_name": "ag_news",
        "hf_config": None,
        "text_keys": ["text"],
        "label_key": "label",
        "num_classes": 4,
        "split_train": "train",
        "split_val": "test",
    },
    "trec": {
        "hf_name": "trec",
        "hf_config": None,
        "text_keys": ["text"],
        "label_key": "coarse_label",
        "num_classes": 6,
        "split_train": "train",
        "split_val": "test",
    },
}

cfg = TASK_CONFIG[args.task]
print(f"Loading {args.task} ({cfg['num_classes']} classes)...")

raw = load_dataset(cfg["hf_name"], cfg["hf_config"]) if cfg["hf_config"] \
      else load_dataset(cfg["hf_name"])

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


def tokenize_split(hf_split):
    """Pre-tokenize all examples in a split. Returns tensors + labels list."""
    texts = hf_split[cfg["text_keys"][0]]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    labels = list(hf_split[cfg["label_key"]])
    return enc, labels


print("Tokenizing train split...")
train_enc, train_labels_clean = tokenize_split(raw[cfg["split_train"]])
print("Tokenizing val split...")
val_enc, val_labels = tokenize_split(raw[cfg["split_val"]])

print(f"Train size: {len(train_labels_clean)} | Val size: {len(val_labels)}")


class TextClsDataset(Dataset):
    """
    Wraps pre-tokenized encodings + labels. Labels are stored externally so
    we can swap noisy / clean labels without re-tokenizing.
    """
    def __init__(self, enc, labels):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids", None)
        self.labels = labels  # list[int] or tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item, int(self.labels[idx])


# Inject label noise into training labels (uniform random flip to a different
# class). This mirrors _NoisyLabelDataset from the CIFAR script.
def make_noisy_labels(clean_labels, noise_ratio, num_classes, seed):
    rng_noise = np.random.default_rng(seed)
    n = len(clean_labels)
    noisy = list(clean_labels)
    noisy_idx = rng_noise.choice(n, size=int(noise_ratio * n), replace=False)
    for i in noisy_idx:
        orig = noisy[i]
        choices = [c for c in range(num_classes) if c != orig]
        noisy[i] = int(rng_noise.choice(choices))
    return noisy


train_labels_noisy = make_noisy_labels(
    train_labels_clean, args.noise_ratio, cfg["num_classes"], seed=args.seed)

train_set = TextClsDataset(train_enc, train_labels_noisy)
val_set = TextClsDataset(val_enc, val_labels)

# Fixed subset of the training set for TPV estimation. TPV does not actually
# use the labels (only the input -> logit mapping), but we keep them aligned
# so the dataset wrapper is consistent.
rng = np.random.default_rng(args.seed)
tpv_indices = rng.choice(len(train_set), size=min(args.tpv_subset_size,
                                                  len(train_set)),
                         replace=False)
tpv_subset = Subset(train_set, tpv_indices.tolist())

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, drop_last=False)
val_loader = DataLoader(
    val_set, batch_size=args.tpv_batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

print(f"Train ({args.noise_ratio:.0%} label noise), TPV subset size: "
      f"{len(tpv_subset)}.")


# ----------------------------
# Model factory & param-group helper
# ----------------------------
def make_model():
    return AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=cfg["num_classes"])


def adamw_param_groups(model, weight_decay):
    """Standard BERT-style param groups: no weight decay on bias/LayerNorm."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(nd in n for nd in no_decay)]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_forward_logits(model, batch):
    """Forward pass that returns logits for either main or noisy stage."""
    out = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items()})
    return out.logits


# ----------------------------
# Standard accuracy eval
# ----------------------------
@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for batch, y in loader:
        y = y.to(device, non_blocking=True)
        logits = model_forward_logits(model, batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ----------------------------
# TPV estimation (noisy-logit fine-tuning)
# ----------------------------
class _LogitNoiseDataset(Dataset):
    """
    Yields (batch_inputs, noisy_logit, teacher_logit, label) for the TPV subset.
    Indexed identically to the underlying Subset.
    """
    def __init__(self, subset, teacher_logits, noisy_logits, labels):
        self.subset = subset
        self.teacher_logits = teacher_logits
        self.noisy_logits = noisy_logits
        self.labels = labels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        batch_inputs, _y = self.subset[idx]
        return (batch_inputs,
                self.noisy_logits[idx],
                self.teacher_logits[idx],
                self.labels[idx])


@torch.no_grad()
def compute_teacher_logits(model, subset, batch_size, num_classes):
    """Compute reference logits f*(x_i) and labels on the TPV subset."""
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    N = len(subset)
    teacher_logits = torch.empty((N, num_classes), dtype=torch.float32)
    labels = torch.empty((N,), dtype=torch.long)
    model.eval()
    offset = 0
    for batch, y in loader:
        bsz = y.size(0)
        z = model_forward_logits(model, batch)
        teacher_logits[offset:offset+bsz].copy_(z.cpu())
        labels[offset:offset+bsz].copy_(y)
        offset += bsz
    return teacher_logits, labels


def _train_noisy(model, loader, max_epochs, lr):
    """
    MSE fine-tune on noisy logit targets. Runs in eval() mode to freeze
    dropout (BERT's analog of frozen BN running stats in the CIFAR script).
    """
    model.eval()  # freeze dropout during fine-tuning
    optimizer = optim.AdamW(adamw_param_groups(model, weight_decay=0.0),
                            lr=lr)
    criterion = nn.MSELoss()
    for _ in range(max_epochs):
        for batch, noisy_z, _teacher_z, _y in loader:
            noisy_z = noisy_z.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model_forward_logits(model, batch)
            loss = criterion(logits, noisy_z)
            if not torch.isfinite(loss):
                return False
            loss.backward()
            optimizer.step()
    return True


def estimate_tpv(base_state_dict, num_classes, R, noise_std,
                 max_epochs_noisy, lr_noisy, train_batch_size, eval_batch_size,
                 model_factory):
    """
    Estimate empirical training-set TPV at the current model weights.

    Returns: float, mean squared deviation of fine-tuned logits from teacher
             logits, averaged over the TPV subset and R runs.
    """
    # 1. Reference (teacher) logits on TPV subset.
    teacher_model = model_factory().to(device)
    teacher_model.load_state_dict(base_state_dict)
    teacher_model.eval()
    teacher_logits, labels = compute_teacher_logits(
        teacher_model, tpv_subset, batch_size=eval_batch_size,
        num_classes=num_classes)
    del teacher_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Adaptive noise scale: sigma * mean(|f*|).
    if ADAPTIVE_NOISE_SCALING:
        logit_scale = teacher_logits.abs().mean().item()
    else:
        logit_scale = 1.0

    # Deterministic shuffle order across runs so only logit noise varies.
    LOADER_SEED = 12345

    total_sqdiff = 0.0
    n_subset = teacher_logits.shape[0]

    for r in range(R):
        eps = torch.randn_like(teacher_logits) * noise_std * logit_scale
        noisy_logits = teacher_logits + eps

        run_dataset = _LogitNoiseDataset(
            subset=tpv_subset,
            teacher_logits=teacher_logits,
            noisy_logits=noisy_logits,
            labels=labels,
        )
        gen = torch.Generator(); gen.manual_seed(LOADER_SEED)
        run_loader = DataLoader(
            run_dataset, batch_size=train_batch_size, shuffle=True,
            generator=gen, num_workers=args.num_workers, pin_memory=True)

        # Re-instantiate model from current checkpoint for this run.
        model = model_factory().to(device)
        model.load_state_dict(base_state_dict)

        ok = _train_noisy(model, run_loader,
                          max_epochs=max_epochs_noisy, lr=lr_noisy)
        if not ok:
            print(f"      [WARNING] TPV run {r} diverged, skipping.")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Squared deviation from teacher logits over the TPV subset.
        eval_loader = DataLoader(
            run_dataset, batch_size=eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        run_sq = 0.0
        model.eval()
        with torch.no_grad():
            for batch, _noisy_z, teacher_z, _y in eval_loader:
                teacher_z = teacher_z.to(device, non_blocking=True)
                z = model_forward_logits(model, batch)
                run_sq += torch.sum((z - teacher_z) ** 2).item()
        total_sqdiff += run_sq

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    empirical_tpv = total_sqdiff / (R * n_subset)
    return empirical_tpv


# ----------------------------
# Build model + optimizer for the main training run
# ----------------------------
model = make_model().to(device)
print(f"Model: {args.model_name} (params={count_params(model):,})")

optimizer = optim.AdamW(adamw_param_groups(model, weight_decay=args.wd),
                        lr=args.lr)
total_steps = len(train_loader) * args.epochs
warmup_steps = int(args.warmup_ratio * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss(label_smoothing=args.labelsmooth)

results_path = (
    f"results/tpv_trajectory_bert_{args.task}_seed{args.seed}_"
    f"noise_ratio-{args.noise_ratio}_labelsmooth-{args.labelsmooth}_epoch-{args.epochs}.pkl"
)


# ----------------------------
# Training loop with periodic TPV
# ----------------------------
epoch_log     = []
train_acc_log = []
val_acc_log   = []
tpv_epochs    = []
tpv_log       = []


def maybe_compute_tpv(epoch):
    """Compute TPV at this epoch and append to logs."""
    print(f"  [TPV] Estimating training-set TPV at epoch {epoch}...")
    base_state = copy.deepcopy(model.state_dict())
    tpv_value = estimate_tpv(
        base_state_dict=base_state,
        num_classes=cfg["num_classes"],
        R=args.R,
        noise_std=args.noise_std,
        max_epochs_noisy=args.max_epochs_noisy,
        lr_noisy=args.lr_noisy,
        train_batch_size=args.tpv_batch_size,
        eval_batch_size=args.tpv_batch_size,
        model_factory=make_model,
    )
    tpv_epochs.append(epoch)
    tpv_log.append(tpv_value)
    print(f"  [TPV] epoch {epoch}: TPV(train) = {tpv_value:.4e}")


# Compute TPV at epoch 0 (initial weights) for reference. Note: for BERT-Small
# this is the *pretrained* backbone with a freshly initialized classifier head,
# so logits are dominated by the random head and TPV at epoch 0 may be small
# in absolute terms; we keep it for plotting consistency with the CIFAR setup.
print("\n=== Computing TPV at epoch 0 (initialization) ===")
maybe_compute_tpv(epoch=0)

tr_acc0 = eval_accuracy(model, train_loader)
va_acc0 = eval_accuracy(model, val_loader)
epoch_log.append(0)
train_acc_log.append(tr_acc0)
val_acc_log.append(va_acc0)
print(f"  Epoch 0: train_acc={tr_acc0:.4f}, val_acc={va_acc0:.4f}")

print("\n=== Training ===")
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader,
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False)
    for batch, y in pbar:
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model_forward_logits(model, batch)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / max(n_batches,1):.3f}"})

    train_acc = eval_accuracy(model, train_loader)
    val_acc   = eval_accuracy(model, val_loader)
    epoch_log.append(epoch)
    train_acc_log.append(train_acc)
    val_acc_log.append(val_acc)
    print(f"  Epoch {epoch:3d} | lr={scheduler.get_last_lr()[0]:.2e} | "
          f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    if epoch % args.tpv_every == 0 or epoch == args.epochs:
        maybe_compute_tpv(epoch=epoch)

        # Persist intermediate logs.
        with open(results_path, "wb") as f:
            pkl.dump(dict(
                epoch_log=epoch_log,
                train_acc_log=train_acc_log,
                val_acc_log=val_acc_log,
                tpv_epochs=tpv_epochs,
                tpv_log=tpv_log,
                args=vars(args),
            ), f)

print(f"\nSaved logs to {results_path}")

# Reload for plotting (matches CIFAR script style).
with open(results_path, "rb") as f:
    data = pkl.load(f)
    epoch_log = data["epoch_log"]
    train_acc_log = data["train_acc_log"]
    val_acc_log = data["val_acc_log"]
    tpv_epochs = data["tpv_epochs"]
    tpv_log = data["tpv_log"]


# ----------------------------
# Plot: dual-axis (accuracy on left, TPV on right)
# ----------------------------
LABEL_FONTSIZE  = 16
TICK_FONTSIZE   = 13
LEGEND_FONTSIZE = 10

fig, ax_acc = plt.subplots(figsize=(6, 3))
ax_tpv = ax_acc.twinx()

l1, = ax_acc.plot(epoch_log, train_acc_log, "-",  color="tab:blue",
                  label="Train accuracy", linewidth=1.8)
l2, = ax_acc.plot(epoch_log, val_acc_log,   "-",  color="tab:orange",
                  label="Validation accuracy", linewidth=1.8)
l3, = ax_tpv.plot(tpv_epochs, tpv_log,      "-o", color="tab:green",
                  label="Training-set TPV", linewidth=1.8, markersize=5)

ax_acc.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
ax_acc.set_ylabel("Accuracy", fontsize=LABEL_FONTSIZE, color="black")
ax_tpv.set_ylabel("Training-set TPV", fontsize=LABEL_FONTSIZE,
                  color="tab:green")
ax_acc.tick_params(labelsize=TICK_FONTSIZE)
ax_tpv.tick_params(labelsize=TICK_FONTSIZE, colors="tab:green")
ax_tpv.set_yscale("log")

# Mark the epoch with maximum TPV.
if len(tpv_log) > 0:
    argmax_tpv_idx = int(np.argmax(tpv_log))
    l4 = ax_acc.axvline(tpv_epochs[argmax_tpv_idx], linestyle="--",
                        color="tab:green", alpha=0.4,
                        label="argmax_epoch TPV")
    lines  = [l1, l2, l3, l4]
else:
    lines = [l1, l2, l3]

labels = [ln.get_label() for ln in lines]
ax_acc.legend(lines, labels, fontsize=LEGEND_FONTSIZE,
              loc="lower center", framealpha=0.4)

plt.tight_layout()
ax_acc.grid()
plot_path = (
    f"results/plots/tpv_trajectory_bert_{args.task}_seed{args.seed}_"
    f"noise_ratio-{args.noise_ratio}_labelsmooth-{args.labelsmooth}_epoch-{args.epochs}.pdf"
)
plt.savefig(plot_path, bbox_inches="tight")
plt.show()
print(f"Saved plot to: {plot_path}")
