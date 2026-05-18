# TPV: Parameter Perturbations Through the Lens of Test Prediction Variance

Official repository for the paper **"TPV: Parameter Perturbations Through the Lens of Test Prediction Variance"**, accepted at **ICML 2026**.

[[Paper (arXiv)](https://arxiv.org/abs/2512.11089)] &nbsp;|&nbsp; [BibTeX](#citation)

---

## Overview

**Test Prediction Variance (TPV)** is a unified framework for analysing how parameter perturbations—label noise, SGD noise, quantization, and pruning—affect a trained model's outputs. The key identity,

```
TPV(w) ≈ Tr(H_eff C)
```

separates the geometry of the trained model (*H_eff*, the second moment of the output-parameter Jacobian) from the perturbation covariance (*C*), placing all major perturbation sources under one lens. This repo contains the code to reproduce every experiment in the paper.

---

## Installation

```bash
git clone https://github.com/devansharpit/TPV.git
cd TPV
pip install -r requirements.txt
```

---

## Repository Structure

```
TPV/
├── engine/                   # Core model and dataset library
│   ├── models/               # CIFAR, ImageNet, and graph architectures
│   └── utils/                # Datasets, metrics, and evaluation utilities
├── investigation/            # Section 5 — Empirical Verification experiments
├── applications/             # Section 6 — Applications experiments
├── example_tpv_resnet.py     # Quick-Start Example for TPV usage
├── imagenet_dataloader.py    # ImageNet data-loading utility (shared)
├── presets.py                # Training presets (shared)
├── registry.py               # Model registry (shared)
├── requirements.txt
└── LICENSE
```

> **All commands below should be run from the repository root.**

---

## Quick-Start Example

### What is TPV?

**Test Prediction Variance (TPV)** measures how much a trained model's predictions change when its parameters are perturbed. Formally (Eq. 2 in the paper), it is the expected squared change in outputs under a zero-mean parameter perturbation *δu* around a fixed trained solution *w\**:

```
TPV := E_{x, δu} [ ‖f_{w*+δu}(x) − f_{w*}(x)‖² ]
```

Via a first-order Taylor expansion around *w\**, this reduces to an efficient **trace form**:

```
TPV(w) ≈ Tr(H_eff C)
```

where *H_eff* is the second moment of the output-parameter Jacobian (a label-free geometric factor capturing the model's sensitivity directions) and *C* is the perturbation covariance matrix that encodes the specific noise mechanism. Crucially, this factorisation means all major perturbation sources — SGD noise, quantization, label noise, pruning — are handled by the same formula; only *C* changes.

### TPV Stability

The central theoretical result of the paper (Theorem 3.1) is **TPV stability**: in the over-parameterised limit, the TPV computed on the *training set* converges to the TPV computed on the *test set*, irrespective of the model's generalisation performance. Concretely:

```
|TPV(w*; X_train) − TPV(w*; X_test)| ≤ c₁ · Tr(C)
```

This means training-set TPV is a reliable, label-free estimator of test-time prediction sensitivity. Stability is decoupled from generalisation and holds empirically even at relatively modest network widths, only breaking down when the number of training samples is very small or the induced perturbations are unusually large.

### Supported Noise Sources

| Noise source | Perturbation covariance *C* |
|---|---|
| **SGD noise** | Stationary weight fluctuations from mini-batch SGD; `C_SGD ≈ (lr / (2·batch_size)) · ∇²L` |
| **Quantization noise** | Uniform *b*-bit quantization; per-weight variance `σ_q² = δ²/12`, where `δ = weight_range / (2^b − 1)` |
| **Label noise** | Gaussian perturbations of training targets; sensitivity governed by the Jacobian's singular-value structure (directions with large `B_ii` and small `s_i` are most vulnerable) |

### Running the Example

`example_tpv_resnet.py` provides a self-contained demonstration of TPV stability across all three noise sources. It loads a pretrained ResNet-20 on CIFAR-10 (via the `chenyaofo` hub) and draws 2 000-sample train and test subsets. For each noise source it computes TPV two ways:

- **Trace form** — uses a Hutchinson estimator (50 Rademacher vectors) to evaluate `Tr(H_eff)` analytically
- **Empirical variant** — SGD: averages over trajectory snapshots after a burn-in; Quantization: Monte Carlo over 50 random quantization draws; Label noise: repeated fine-tuning over *R* = 5 noisy-target realisations

The script prints a summary table of `TPV_train` vs `TPV_test` for all five (source, estimator) combinations and the train/test stability ratio, which should be close to 1.

```bash
python example_tpv_resnet.py
```

**Example output:**

```
========================================================================================
  TPV Summary — cifar10_resnet20 (pretrained on CIFAR-10)
========================================================================================
  Perturbation  Estimator          Train TPV      Test TPV  Notes
  --------------------------------------------------------------------------------------
  SGD noise     trace form        1.1924e+00    1.3405e+00  lr=0.001, batch=128
  SGD noise     empirical         6.0085e-01    5.7196e-01  sgd_steps=200, snaps=16
  Quantization  trace form        1.5959e+00    1.7831e+00  8-bit uniform
  Quantization  empirical         1.6877e+00    1.7390e+00  8-bit uniform, n_runs=50
  Label noise   empirical         1.4289e-02    1.2434e-02  noise_std=0.1, R=5
========================================================================================

  Train/Test stability (TPV_train / TPV_test; should ideally be close to 1):
    SGD noise      trace form      0.889
    SGD noise      empirical       1.051
    Quantization   trace form      0.895
    Quantization   empirical       0.970
    Label noise    empirical       1.149
```

---

## Reproducing Experiments

### Section 5 — Empirical Verification

---

#### Section 5.1 &nbsp; TPV Stability

**Figures 1 & 2 — TPV stability on synthetic data (vary width and n_train)**

Runs 324 configurations spanning dataset type, input dimension, network width, depth, and training-set size under label-noise and SGD-noise perturbations. Produces two scatter plots (TPV_train vs TPV_test): one varying network width (Fig. 1) and one varying the number of training samples (Fig. 2).

```bash
python investigation/tpv_trace_synth_universal_scatter.py
```

---

**Figure 3 — TPV stability on CIFAR-10 (vary network width)**

Replicates the synthetic scatter experiment on CIFAR-10 using MobileNetV2 at different width multipliers, confirming that TPV stability holds for real vision architectures.

```bash
python investigation/tpv_cifar_universal_scatter_vary_w.py --dataset c10 --savefile tpv_cifar10_width_sweep
```

---

**Figure 12 (Appendix G) — TPV stability on CIFAR-10 (vary n_train)**

Same scatter experiment on CIFAR-10, now varying the number of training samples rather than width.

```bash
python investigation/tpv_cifar_universal_scatter_vary_n_train.py --dataset c10 --savefile tpv_cifar10_vary_n_train
```

---

**Figure 13 (Appendix G) — TPV stability on CIFAR-100 (vary network width)**

```bash
python investigation/tpv_cifar_universal_scatter_vary_w.py --dataset c100 --savefile tpv_cifar100_width_sweep
```

---

**Figure 14 (Appendix G) — TPV stability on CIFAR-100 (vary n_train)**

```bash
python investigation/tpv_cifar_universal_scatter_vary_n_train.py --dataset c100 --savefile tpv_cifar100_vary_n_train
```

---

#### Section 5.2 &nbsp; Model Width and Label Noise TPV

**Figures 4 & 5 — Empirical and theoretical TPV under label noise on synthetic data**

Trains MLPs of varying widths on a synthetic Gaussian linear-teacher task and computes both empirical and closed-form theoretical TPV under label noise. Also generates Figure 16 (Appendix G), which plots the generalization gap and T_base vs. width.

```bash
python investigation/tpv_label_noise_synth_data.py
```

---

**Figure 6 — Empirical TPV under logit noise on CIFAR-100 (vary width)**

Measures how label-noise TPV changes with network width on CIFAR-100, showing that both TPV estimates decrease with width and track the reference model's clean test loss.

```bash
python investigation/tpv_label_noise_cifar.py --dataset cifar100
```

---

**Figure 15 (Appendix G) — Empirical TPV under logit noise on CIFAR-10 (vary width)**

```bash
python investigation/tpv_label_noise_cifar.py --dataset cifar10
```

---

#### Section 5.3 &nbsp; Label Noise TPV and Generalization

**Figure 7 — TPV vs. test loss on CIFAR-10 across architecture sizes and regularization levels (label smoothing)**

Sweeps label-smoothing strength across MLP architectures of different sizes on CIFAR-10. U-shape emerges: in the high training loss regime, lower TPV corresponds to higher train and test loss due to underfitting; in the low training loss regime, higher TPV corresponds to lower train loss and higher test loss due to overfitting showing that test-set label-noise TPV is correlated with test loss in the low training loss regime.

```bash
python investigation/tpv_test_loss_cifar_single_reg.py --reg_type labelsmooth
```

---

**Figure 17 (Appendix G) — TPV vs. test loss on CIFAR-10 across architecture sizes and regularization levels (dropout)**

Similar conclusions as the label smoothing experiment above.

```bash
python investigation/tpv_test_loss_cifar_single_reg.py --reg_type dropout
```

---

**Figure 18 (Appendix G) — TPV vs. test loss on CIFAR-10 across architecture sizes and regularization levels (weight decay)**

Similar conclusions as the label smoothing experiment above.

```bash
python investigation/tpv_test_loss_cifar_single_reg.py --reg_type wd
```

---



**Figure 8 — TPV trajectory on CIFAR-100 with 30% label noise (ResNet-18)**

Tracks TPV epoch-by-epoch during training, revealing its U-shaped relationship with validation accuracy.

```bash
python investigation/tpv_trajectory_cifar100.py --labelsmooth 0.1 --noise_ratio 0.3
```

---

**Figures 19 & 20 (Appendix G) — TPV trajectory for BERT-small (AG News and TREC)**

Demonstrates TPV-based monitoring during BERT fine-tuning under label noise on two NLU tasks, replicating the U-shaped dynamics observed on CIFAR.

```bash
# Figure 19 — AG News
python investigation/tpv_trajectory_bert.py --task ag_news --noise_ratio 0.20 --epochs 50 --lr 1e-4 --labelsmooth 0

# Figure 20 — TREC
python investigation/tpv_trajectory_bert.py --task trec --noise_ratio 0.20 --epochs 50 --lr 1e-4 --labelsmooth 0
```

---

### Section 6 — Applications

---

#### Note for ImageNet Experiments Below

> **Note:** Update the `create_imagenet_dataloaders` function in `imagenet_dataloader.py` to point to your local ImageNet directory before running.

#### Section 6.1 &nbsp; Pruning

**Figures 21–24 — JBR pruning criterion vs. baselines (CIFAR-10, CIFAR-100, ImageNet)**

Benchmarks the JBR (Jacobian-Based Rebalancing) pruning criterion derived from TPV against Jacobian, L1, Taylor, BN Scale, FPGM, WHC, and Random baselines. Code adapted from the official [Optimal Brain Connection](https://github.com/ShaowuChen/Optimal_Brain_Connection) repository.

```bash
# ResNet-56 on CIFAR-10 (Figs 21, 23)
python applications/benchmark_pruning.py --model resnet56_cifar10 \
    --repeats 5 --N_batchs 50 --global_pruning --pruning_ratio 0.9 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir resnet56_cifar10

# VGG-19 on CIFAR-100 (Figs 21, 23)
python applications/benchmark_pruning.py --model vgg19_bn_cifar100 \
    --repeats 5 --N_batchs 50 --global_pruning --pruning_ratio 0.9 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir vgg19_bn_cifar100

# ResNet-50 on ImageNet (Figs 22, 24)
python applications/benchmark_pruning.py --model resnet50_imagenet \
    --repeats 5 --N_batchs 50 --global_pruning --pruning_ratio 0.5 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir resnet50_imagenet_bs256 --resume

# MobileNet-v2 on ImageNet (Figs 22, 24)
python applications/benchmark_pruning.py --model mobilenet_v2_imagenet \
    --repeats 5 --N_batchs 50 --global_pruning --pruning_ratio 0.5 \
    --iterative_steps 18 \
    --run_criteria "['Jacobian', 'JBR', 'L1', 'Random', 'BN Scale', 'FPGM', 'WHC', 'Taylor']" \
    --save_dir mobilenet_v2_imagenet
```

---

#### Section 6.2 &nbsp; Model Selection

**Figure 9 — In-distribution training recipe selection on CIFAR-10**

Samples five joint regularization configurations (weight decay, dropout, label smoothing) and trains ResNet architectures of different sizes under each, demonstrating that training-set TPV reliably identifies the recipe that generalises best.

```bash
python applications/tpv_cifar_reg_combination.py
```

---

**Figure 10 — Cross-architecture model selection on ImageNet under label noise**

Fine-tunes multiple pretrained ImageNet architectures under label noise and shows that training-set TPV tracks test-set TPV and correlates with generalisation performance, enabling label-free model selection.

```bash
python applications/tpv_label_noise_imagenet.py
```

---

**Figure 25 (Appendix I) — Transfer-learning recipe selection on Oxford Pets**

Fine-tunes four ImageNet-pretrained backbones on Oxford Pets under label noise across multiple joint regularization configurations, showing that training-set TPV selects the best recipe without requiring a validation set.

```bash
python applications/tpv_label_noise_oxford_pets.py
```

---

**Figure 11 — Training-set TPV vs. sharpness for predicting label-noise sensitivity**

Compares Hessian trace (sharpness / SGD-noise TPV) against training-set label-noise TPV as predictors of test-set label-noise sensitivity across models with varying weight decay. Shows that label noise TPV is the reliable signal while sharpness is not.

```bash
python applications/label_noise_sensitivity_label-noise-tvp_vs_sgd-noise-tpv.py
```

## Citation

```bibtex
@inproceedings{arpit2026tpv,
  title     = {TPV: Parameter Perturbations Through the Lens of Test Prediction Variance},
  author    = {Arpit, Devansh},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR}
}
```
